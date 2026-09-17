"""Блок 20: multi-token backbone loss с обучаемой балансировкой offset-весов
(uncertainty weighting, Kendall, Gal & Cipolla, 2018) вместо фиксированных
1.0/0.25/0.125 (блок 17) и вместо warmup/detach-расписания (по условию
задачи запрещено даже временно резать backprop через часть весов — веса
между задачами должны быть обучаемыми, а не результатом ручного расписания).

Для каждого offset заводится обучаемый скаляр log_sigma_i = log(sigma_i^2).
Итоговый loss:

    L = Σ_i [ exp(-log_sigma_i) · L_i + log_sigma_i ]

exp(-log_sigma_i) — эффективный вес задачи i (обучается вместе со всеми
остальными параметрами обычным backprop, без отдельного расписания).
log_sigma_i — регуляризатор: без него сеть могла бы тривиально занулить вес
любой задачи, просто разогнав log_sigma_i к +inf, ничего не улучшая в L_i
самой. В равновесии (если бы оптимизировался только этот член)
exp(-log_sigma_i) ≈ 1/L_i — более трудная (с большим loss) задача
автоматически получает меньший вес, простая — больший.

"Не считать targets за концом документа" (PLAN.md §5.3): если между текущей
позицией и целью на offset>1 встретился EOS, target маскируется (ignore_index).
"""

import torch
import torch.nn as nn

from src.sid.chunked import chunked_readout_loss

OFFSETS = (1, 2, 4)  # PLAN.md §5.3: next-token + 2 + 4 позиции вперёд
MAX_OFFSET = max(OFFSETS)


class MultiTokenHeads(nn.Module):
    """Линейные проекции для offset=2,4 (offset=1 — обычный readout(h0) без
    проекции) + обучаемые log_sigma, по одному на offset."""

    def __init__(self, n_embd):
        super().__init__()
        self.proj = nn.ModuleDict({
            str(offset): nn.Linear(n_embd, n_embd) for offset in OFFSETS if offset != 1
        })
        for layer in self.proj.values():
            torch.nn.init.normal_(layer.weight, mean=0.0, std=0.02)
            torch.nn.init.zeros_(layer.bias)
        # log_sigma=0 в старте -> exp(-log_sigma)=1 для всех -> равные веса на
        # старте (не 1.0/0.25/0.125 из блока 17 — теперь это точка отсчёта,
        # а не финальное решение), дальше сеть подбирает пропорции сама.
        self.log_sigma = nn.Parameter(torch.zeros(len(OFFSETS)))


def multi_token_loss(model, heads, h0, raw_tokens, eos_id):
    """Возвращает (total_loss, {offset: L_i.item()}, {offset: exp(-log_sigma_i).item()})."""
    block_size = h0.size(1)
    assert raw_tokens.size(1) == block_size + MAX_OFFSET

    is_eos = (raw_tokens == eos_id)
    cum = is_eos.cumsum(dim=1)

    total = h0.new_zeros(())
    per_offset_loss = {}
    per_offset_weight = {}
    for idx, offset in enumerate(OFFSETS):
        target = raw_tokens[:, offset:offset + block_size]
        proj = None

        if offset == 1:
            valid = None  # offset=1 маскировать нечего: eos сам по себе валидный target
        else:
            cum_hi = cum[:, offset - 1: offset - 1 + block_size]
            cum_lo = cum[:, 0:block_size]
            valid = (cum_hi - cum_lo) == 0
            proj = heads.proj[str(offset)]

        target = target if valid is None else target.masked_fill(~valid, -1)
        loss_o = chunked_readout_loss(model, h0, target, ignore_index=-1, proj=proj)

        log_sigma_i = heads.log_sigma[idx]
        weight = torch.exp(-log_sigma_i)
        total = total + weight * loss_o + log_sigma_i

        per_offset_loss[offset] = loss_o.item()
        per_offset_weight[offset] = weight.item()

    return total, per_offset_loss, per_offset_weight


class BoostWeights(nn.Module):
    """Обучаемый скаляр alpha_i на вклад каждого верхнего блока в сумму
    логитов (sid/chunked.py::chunked_boost_loss). Исправление находки: без
    веса свежеинициализированные блоки добавляют плохо откалиброванные
    логиты прямо в сумму, и ошибка КОМПАУНДИРУЕТСЯ с глубиной (val_ppl росла
    124->131->146->172->212->273 вместо падения) — см. memory.md.

    alpha инициализируется в ~0 (тот же принцип, что AdaLN-Zero в
    диффузионных трансформерах): блок не может испортить ансамбль, пока не
    докажет пользу собственным градиентом (градиент по alpha_i ненулевой
    даже при alpha_i=0, т.к. d(loss)/d(alpha_i) зависит от logits_i, а не от
    текущего значения alpha_i)."""

    def __init__(self, num_blocks, init=0.01):
        super().__init__()
        self.alpha = nn.Parameter(torch.full((num_blocks,), init))
