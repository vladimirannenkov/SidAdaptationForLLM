
import torch
import torch.nn.functional as F


def embed(model, idx):
    """idx (B,T) -> (B,T,C): embeddings + dropout, без единого Transformer-слоя."""
    _, t = idx.size()
    pos = torch.arange(0, t, dtype=torch.long, device=idx.device)
    return model.transformer.drop(model.transformer.wte(idx) + model.transformer.wpe(pos))


def forward_range(model, x, start, end):
    """Проход по transformer.h[start:end] над уже готовым hidden state x.
    start==end -> x не меняется (пустой диапазон, валиден: например, backbone
    при k=0 в SID-разбиении с k=n_layer с другой стороны, см. блок 15)."""
    for layer in model.transformer.h[start:end]:
        x = layer(x)
    return x


def readout(model, x):
    """Общий readout: ln_f + lm_head. Тот же модуль, что и в GPT.forward —
    не копия, а вызов тех же submodules."""
    return model.lm_head(model.transformer.ln_f(x))


def frozen_readout(model, x):
    """Как readout(model, x) выше, но с detach()-нутыми весами ln_f/lm_head:
    градиент течёт в x (скрытое состояние блока), НО НЕ накапливается на
    параметрах readout. Используется в абляциях "заморозить readout для всех
    блоков кроме последнего" (train_sid_f.py --freeze-readout except-last,
    train_sid_p_boost.py --freeze-readout except-last) — значения совпадают с
    обычным readout (те же веса), отличие только в том, куда течёт градиент."""
    ln = model.transformer.ln_f
    bias = ln.bias.detach() if ln.bias is not None else None
    normed = F.layer_norm(x, ln.weight.shape, ln.weight.detach(), bias, 1e-5)
    return F.linear(normed, model.lm_head.weight.detach())
