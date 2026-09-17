
import torch


def partition_parameters(model, k, upper_block_size=1):
    """upper_block_size — сколько слоёв (после k) объединяются в один
    "верхний блок" (по умолчанию 1 — старое поведение, каждый слой сам себе
    блок; не меняет вызовы без этого аргумента). Например k=4,
    upper_block_size=2 на 12-слойной модели даёт 4 верхних блока по 2 слоя
    каждый (слои 4-5, 6-7, 8-9, 10-11), а не 8 блоков по 1 слою."""

    backbone = list(model.transformer.wte.parameters()) + list(model.transformer.wpe.parameters())
    for layer in model.transformer.h[:k]:
        backbone += list(layer.parameters())

    readout = list(model.transformer.ln_f.parameters()) + list(model.lm_head.parameters())

    n_layer = len(model.transformer.h)
    blocks = []
    for start in range(k, n_layer, upper_block_size):
        end = min(start + upper_block_size, n_layer)
        params = []
        for layer in model.transformer.h[start:end]:
            params += list(layer.parameters())
        blocks.append(params)

    return {"backbone": backbone, "readout": readout, "blocks": blocks}


def configure_sid_optimizers(model, k, weight_decay, learning_rate, betas, upper_block_size=1):
    groups = partition_parameters(model, k, upper_block_size)

    def make_optimizer(params):
        decay = [p for p in params if p.requires_grad and p.dim() >= 2]
        nodecay = [p for p in params if p.requires_grad and p.dim() < 2]
        return torch.optim.AdamW(
            [{"params": decay, "weight_decay": weight_decay},
             {"params": nodecay, "weight_decay": 0.0}],
            lr=learning_rate, betas=betas, foreach=False,
        )

    return {
        "backbone": make_optimizer(groups["backbone"]),
        "readout": make_optimizer(groups["readout"]),
        "blocks": [make_optimizer(block_params) for block_params in groups["blocks"]],
    }
