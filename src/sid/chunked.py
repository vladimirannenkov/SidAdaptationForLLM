"""Блок 19: порционный (chunked) readout+CE — PLAN.md §5/§6.

Обычный путь: readout(model, h) на всей последовательности сразу даёт
(B, T, V) тензор логитов — при V=16384, T=256 это ~1 ГиБ на один такой
тензор в fp32. multi_token_loss (блок 17) считает три таких тензора подряд
(offset 1,2,4) — теоретически можно держать все три в памяти одновременно
(если PyTorch не успел освободить предыдущий до создания следующего).

chunked_readout_loss обрабатывает T кусками по CHUNK_SIZE позиций: в моменте
живёт только (B, chunk_size, V) — на порядок меньше, а итоговое число
(среднее CE по валидным токенам) математически то же самое, что и при
подсчёте на всей последовательности разом (проверено в tools/check_chunked.py).
"""

import torch
import torch.nn.functional as F

from src.sid.forward import readout
from src.sid.newton import apply_trust_region, descent_indicator, fix_minus_break_indicator, newton_quad_per_token

CHUNK_SIZE = 64  # T=256 / 64 = 4 куска; из (B,256,V) в моменте живёт только (B,64,V)


def chunked_readout_loss(model, h, target, ignore_index, proj=None, chunk_size=CHUNK_SIZE):
    """CE(target, readout(proj(h))), посчитанный кусками по T, без материализации
    (B, T, V) целиком. proj — опциональная маленькая проекция (для offset=2,4,
    см. блок 17); None -> readout(h) без проекции (offset=1)."""
    T = h.size(1)
    total_loss = h.new_zeros(())
    total_count = 0
    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        h_chunk = h[:, start:end]
        if proj is not None:
            h_chunk = proj(h_chunk)
        logits_chunk = readout(model, h_chunk)  # (B, chunk, V) — только этот кусок в памяти
        target_chunk = target[:, start:end]

        count = int((target_chunk != ignore_index).sum().item())
        if count == 0:
            continue  # весь кусок замаскирован (например, конец короткого документа)
        loss_chunk = F.cross_entropy(
            logits_chunk.reshape(-1, logits_chunk.size(-1)), target_chunk.reshape(-1),
            ignore_index=ignore_index, reduction="sum",
        )
        total_loss = total_loss + loss_chunk
        total_count += count

    return total_loss / max(total_count, 1)


def chunked_consistency_loss(model, h_prev, h_cur, target, tau, lam, chunk_size=CHUNK_SIZE):
    """Блок 22 (правка): CE(target, readout(h_cur)) + lam*tau^2*KL(sg(softmax(
    readout(h_prev)/tau)) || softmax(readout(h_cur)/tau)) — PLAN.md §5.1,
    посчитанный кусками по T, как chunked_readout_loss выше. Для KL нужны ДВА
    полных (B,T,V) тензора одновременно (p_cur и p_prev) — без чанкинга это
    вдвое хуже по памяти, чем даже обычный CE; здесь оба чанкуются синхронно,
    поэтому в моменте живёт только по одному (B,chunk,V) c каждой стороны.

    readout(h_prev) пересчитывается заново под no_grad на каждом чанке вместо
    хранения полного prev_probs — небольшой повторный matmul вместо памяти
    под целый (B,T,V) тензор.
    """
    T = h_cur.size(1)
    total_ce = h_cur.new_zeros(())
    total_kl = h_cur.new_zeros(())
    total_tokens = 0
    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        target_chunk = target[:, start:end]
        n = target_chunk.numel()

        logits_cur = readout(model, h_cur[:, start:end])
        total_ce = total_ce + F.cross_entropy(
            logits_cur.reshape(-1, logits_cur.size(-1)), target_chunk.reshape(-1), reduction="sum")

        with torch.no_grad():
            logits_prev = readout(model, h_prev[:, start:end])
            prev_probs = F.softmax(logits_prev / tau, dim=-1)
        log_p_cur_tau = F.log_softmax(logits_cur / tau, dim=-1)
        total_kl = total_kl + F.kl_div(
            log_p_cur_tau.reshape(-1, logits_cur.size(-1)), prev_probs.reshape(-1, logits_cur.size(-1)),
            reduction="sum")

        total_tokens += n

    ce = total_ce / total_tokens
    kl = total_kl / total_tokens
    return ce + lam * (tau ** 2) * kl, ce, kl


def chunked_boost_loss(model, hidden_history, alpha_history, h_i, alpha_i, target, chunk_size=CHUNK_SIZE,
                        own_readout_fn=None):
    """Boosting-стиль вместо consistency-KL (адаптация BoostResNet/gradient
    boosting в logit-space — см. memory.md, "подходы из смежных областей").

    Проблема, которую это решает: обычный consistency-KL (chunked_consistency_loss
    выше) штрафует блок за НЕСОГЛАСИЕ с предыдущей глубиной — то есть буквально
    поощряет избыточность (blocks сходятся к почти одинаковым предсказаниям,
    см. val/block*_kl -> ~0 в SID-S-blocks/SID-P пилотах). "Synergistic" в SID
    подразумевает обратное: каждый блок должен вносить ДОПОЛНИТЕЛЬНУЮ информацию.

    Идея (BoostResNet, arXiv:1706.04964: ResNet = телескопическая сумма слабых
    обучающихся; GrowNet, arXiv:2002.07971: gradient boosting в logit-space):
    блок i не предсказывает P(y) самостоятельно, а добавляет ВЗВЕШЕННУЮ
    коррекцию к уже накопленному (под sg) предсказанию всех предыдущих глубин:

        L_i = sg(sum_{j<i} alpha_j · readout(h_j)) + alpha_i · readout(h_i)
        loss_i = CE(target, softmax(L_i))

    ПЕРВАЯ версия (без alpha, i.e. alpha≡1 для всех) была проверена в реальном
    прогоне (train_sid_p_boost.py, RUN_ID 20260920_105453) и провалилась:
    val_ppl по глубине РОСЛА (124->131->146->172->212->273 к шагу 400) вместо
    падения — свежеинициализированные блоки добавляют плохо откалиброванные
    логиты прямо в сумму, и ошибка компаундируется, а не исправляется.
    Исправление: alpha_i — обучаемый скаляр (sid/losses.py::BoostWeights),
    инициализированный около нуля (тот же AdaLN-Zero принцип, что в
    dblocks/conditioning.py) — блок не может испортить ансамбль, пока не
    докажет пользу собственным градиентом (dL/dalpha_i зависит от logits_i,
    не от текущего alpha_i — бутстрап тот же, что для zero-init слоёв).

    alpha_history — alpha_j ПРЕДЫДУЩИХ глубин, передаются уже с .detach()
    вызывающим кодом (или detach делается здесь) — обучает alpha_j только
    СОБСТВЕННЫЙ шаг блока j, не блок i. Для backbone (j=0) alpha обычно
    передаётся как константа 1.0 (уже обученное, доверенное предсказание,
    не "непроверенный" блок) — вызывающий код сам решает.

    hidden_history — список ПОД sg скрытых состояний всех предыдущих глубин
    (backbone + уже обученные блоки), каждое (B,T,C). readout каждого
    пересчитывается заново под no_grad на каждом чанке (как prev_probs в
    chunked_consistency_loss) — не хранится целиком (B,T,V).

    Возвращает loss_i (= CE, для единообразия сигнатуры с chunked_consistency_loss
    вызывающий код использует один и тот же loss и для backward, и для лога) —
    это одновременно и то, что оптимизируется, и то, что значит "ppl ансамбля
    после добавления блока i", то есть кривая по глубине теперь напрямую
    показывает, помогает ли накопление блоков, а не просто локальное качество
    каждого блока по отдельности.

    own_readout_fn — необязательная замена ТОЛЬКО для собственного (с градиентом)
    терма readout(h_i) текущего блока (сигнатура (model, x) -> logits, как у
    sid/forward.py::readout). По умолчанию (None) используется обычный readout.
    Нужно для абляции "заморозить readout для всех блоков кроме последнего":
    вызывающий код (train_sid_p_boost.py) передаёт detach()-нутую версию весов
    ln_f/lm_head для всех блоков, кроме последнего — тогда градиент по-прежнему
    течёт в h_i/alpha_i (сам readout как ФУНКЦИЯ дифференцируем), но НЕ
    накапливается на параметрах readout. cum_logits (история) и так уже
    целиком под no_grad ниже — там заморозка readout не нужна, это никогда
    не давало градиент на readout параметры даже без own_readout_fn.
    """
    own_readout = own_readout_fn if own_readout_fn is not None else readout
    T = h_i.size(1)
    total_ce = h_i.new_zeros(())
    total_tokens = 0
    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        target_chunk = target[:, start:end]
        n = target_chunk.numel()

        with torch.no_grad():
            cum_logits = None
            for h_prev, alpha_prev in zip(hidden_history, alpha_history):
                alpha_prev_val = alpha_prev.detach() if torch.is_tensor(alpha_prev) else alpha_prev
                logits_prev = readout(model, h_prev[:, start:end]) * alpha_prev_val
                cum_logits = logits_prev if cum_logits is None else cum_logits + logits_prev

        logits_i = own_readout(model, h_i[:, start:end]) * alpha_i
        combined = logits_i if cum_logits is None else cum_logits + logits_i

        total_ce = total_ce + F.cross_entropy(
            combined.reshape(-1, combined.size(-1)), target_chunk.reshape(-1), reduction="sum")
        total_tokens += n

    return total_ce / total_tokens


def chunked_boost_newton_loss(model, hidden_history, alpha_history, h_i, alpha_i, target,
                               lam=0.01, beta=1.0, include_curvature=True, trust_region=None,
                               chunk_size=CHUNK_SIZE, own_readout_fn=None):
    """Newton-SID (обсуждение с пользователем 2026-09-21/22, sid/newton.py) —
    та же boosting-конструкция накопленных (под sg) логитов, что и
    chunked_boost_loss выше (cum_logits/alpha_history/alpha_i устроены ровно
    так же), но с ДВУМЯ отличиями:

    1. "Собственный вклад" блока Δz_i = own_readout(h_i) - sg(z_prev) — не
       сразу CE-таргет, а сырая (до умножения на alpha_i) коррекция в
       пространстве логитов, которая используется ДВАЖДЫ: как есть — во
       вспомогательном Newton-квадратичном лоссе (относительно g_prev=p_prev-y
       и H_prev=diag(p_prev)-p_prev·p_prevᵀ, посчитанных из ПРЕДЫДУЩЕГО,
       detached, кумулятивного логита), и умноженная на alpha_i — в реальном
       CE, как и раньше. alpha_i учится ТОЛЬКО через CE (обычная boosting
       логика "насколько большой шаг делать"), Newton-член учит НАПРАВЛЕНИЕ/
       форму коррекции независимо от текущего alpha_i.

    2. Итоговый лосс blocka = CE + beta*newton_quad (вместо чистого CE) —
       newton_quad считается через sid/newton.py::newton_quad_per_token
       (точная curvature-форма для softmax-Hessian, без построения V×V
       матрицы — см. докстринг модуля).

    own_readout_fn/hidden_history/alpha_history — как в chunked_boost_loss
    (переиспользуется тот же паттерн заморозки readout, если понадобится).

    Возвращает (loss, diag), где diag = {"ce", "quad", "norm_delta_z",
    "descent_frac", "fix_minus_break"} — все ТЕНЗОРЫ (0-dim), НЕ .item() —
    вызывающий код сам решает, когда материализовать их в float (только на
    шагах с collect_metrics, как для drift/cka в chunked_boost_loss). Раньше
    здесь были уже посчитанные .item() на каждый вызов (6 блоков x 4
    микробатча x 5 метрик = 120 форс-синхронизаций CUDA НА КАЖДЫЙ шаг) — это
    и оказалось причиной ~20-40x замедления первого pilot-прогона (2026-09-22,
    см. memory.md): .item() блокирует до завершения ВСЕХ висящих CUDA-кернелов,
    убивая асинхронный оverlap между диспетчеризацией со стороны Python и
    реальным исполнением на GPU."""
    own_readout = own_readout_fn if own_readout_fn is not None else readout
    T = h_i.size(1)
    total_ce = h_i.new_zeros(())
    total_quad = h_i.new_zeros(())
    total_norm_delta = h_i.new_zeros(())
    total_descent = h_i.new_zeros(())
    total_fix_minus_break = h_i.new_zeros(())
    total_tokens = 0
    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        target_chunk = target[:, start:end].reshape(-1)
        n = target_chunk.numel()

        with torch.no_grad():
            cum_logits = None
            for h_prev, alpha_prev in zip(hidden_history, alpha_history):
                alpha_prev_val = alpha_prev.detach() if torch.is_tensor(alpha_prev) else alpha_prev
                logits_prev = readout(model, h_prev[:, start:end]) * alpha_prev_val
                cum_logits = logits_prev if cum_logits is None else cum_logits + logits_prev
            z_prev = cum_logits.reshape(n, -1)
            p_prev = F.softmax(z_prev, dim=-1)
            g_prev = p_prev.scatter_add(
                -1, target_chunk.unsqueeze(-1), -torch.ones_like(target_chunk, dtype=p_prev.dtype).unsqueeze(-1))

        z_i_raw = own_readout(model, h_i[:, start:end]).reshape(n, -1)
        delta_z = apply_trust_region(z_i_raw - z_prev, trust_region)

        combined = z_prev + alpha_i * delta_z
        total_ce = total_ce + F.cross_entropy(combined, target_chunk, reduction="sum")
        total_quad = total_quad + newton_quad_per_token(delta_z, p_prev, g_prev, lam, include_curvature).sum()

        with torch.no_grad():
            delta_z_det = delta_z.detach()
            total_norm_delta = total_norm_delta + delta_z_det.norm(dim=-1).sum()
            total_descent = total_descent + descent_indicator(delta_z_det, g_prev).sum()
            p_cur = F.softmax(combined.detach(), dim=-1)
            total_fix_minus_break = total_fix_minus_break + fix_minus_break_indicator(
                p_prev, p_cur, target_chunk).sum()

        total_tokens += n

    ce = total_ce / total_tokens
    quad = total_quad / total_tokens
    loss = ce + beta * quad
    diag = {
        "ce": ce.detach(),
        "quad": quad.detach(),
        "norm_delta_z": total_norm_delta / total_tokens,
        "descent_frac": total_descent / total_tokens,
        "fix_minus_break": total_fix_minus_break / total_tokens,
    }
    return loss, diag
