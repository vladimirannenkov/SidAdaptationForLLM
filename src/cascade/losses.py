"""Chunked objectives for Cascade-SID."""

import torch
import torch.nn.functional as F

from src.sid.forward import readout


def cascade_three_zone_loss(model, h0, previous_innovations, previous_alphas,
                            innovation, alpha, target, correct_threshold=0.2,
                            preserve_threshold=0.8, refine_weight=0.5,
                            preserve_weight=1.0, chunk_size=64):
    """Three-zone loss for one trainable cascade stage.

    All prefix logits are rebuilt under ``no_grad``.  Consequently gradients
    can reach only ``innovation`` and its scalar ``alpha``.
    """
    sums = {name: innovation.new_zeros(()) for name in ("correct", "refine", "preserve", "ce")}
    counts = {name: 0 for name in ("correct", "refine", "preserve", "all")}
    fixes = breaks = 0
    for start in range(0, innovation.size(1), chunk_size):
        end = min(start + chunk_size, innovation.size(1))
        target_chunk = target[:, start:end]
        y = target_chunk.reshape(-1)
        with torch.no_grad():
            z_old = readout(model, h0[:, start:end])
            for u_prev, alpha_prev in zip(previous_innovations, previous_alphas):
                z_old = z_old + alpha_prev.detach() * readout(model, u_prev[:, start:end])
            p_old = F.softmax(z_old, dim=-1)
            p_target = p_old.gather(-1, target_chunk.unsqueeze(-1)).reshape(-1)
            old_top = z_old.argmax(dim=-1).reshape(-1)
        z_new = z_old + alpha * readout(model, innovation[:, start:end])
        flat_new = z_new.reshape(-1, z_new.size(-1))
        ce = F.cross_entropy(flat_new, y, reduction="none")
        correct = p_target < correct_threshold
        refine = (p_target >= correct_threshold) & (p_target < preserve_threshold)
        preserve = p_target >= preserve_threshold
        for name, mask in (("correct", correct), ("refine", refine)):
            n = int(mask.sum().item())
            if n:
                sums[name] = sums[name] + ce[mask].sum()
                counts[name] += n
        n_preserve = int(preserve.sum().item())
        if n_preserve:
            kl = F.kl_div(F.log_softmax(flat_new, dim=-1), p_old.reshape_as(flat_new), reduction="none").sum(-1)
            sums["preserve"] = sums["preserve"] + kl[preserve].sum()
            counts["preserve"] += n_preserve
        sums["ce"] = sums["ce"] + ce.sum()
        counts["all"] += y.numel()
        with torch.no_grad():
            new_top = flat_new.argmax(dim=-1)
            fixes += int(((old_top != y) & (new_top == y)).sum().item())
            breaks += int(((old_top == y) & (new_top != y)).sum().item())

    zero = innovation.new_zeros(())
    l_correct = sums["correct"] / counts["correct"] if counts["correct"] else zero
    l_refine = sums["refine"] / counts["refine"] if counts["refine"] else zero
    l_preserve = sums["preserve"] / counts["preserve"] if counts["preserve"] else zero
    loss = l_correct + refine_weight * l_refine + preserve_weight * l_preserve
    metrics = {
        "correct_ce": l_correct.detach().item(), "refine_ce": l_refine.detach().item(),
        "preserve_kl": l_preserve.detach().item(), "combined_ce": (sums["ce"] / counts["all"]).detach().item(),
        "correct_fraction": counts["correct"] / max(counts["all"], 1),
        "refine_fraction": counts["refine"] / max(counts["all"], 1),
        "preserve_fraction": counts["preserve"] / max(counts["all"], 1),
        "fix_minus_break": (fixes - breaks) / max(counts["all"], 1),
    }
    return loss, metrics
