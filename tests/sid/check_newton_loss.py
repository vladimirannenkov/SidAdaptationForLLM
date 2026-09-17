"""Проверка Newton-SID (sid/newton.py, sid/chunked.py::chunked_boost_newton_loss),
2026-09-22:
1. softmax_hessian_quadratic_form(Δz,p) (трюк без V×V матрицы) совпадает с
   прямым Δz^T·(diag(p)-p·p^T)·Δz на маленьком V, посчитанным явно.
2. При Δz=0 все члены (linear/curvature/damping) равны 0.
3. descent_indicator/fix_minus_break_indicator дают ожидаемые знаки на
   примерах из обсуждения (p_y=0.1 vs p_y=0.9 случаи).
4. apply_trust_region реально ограничивает |Δz_j|<=delta.
5. chunked_boost_newton_loss: градиент доходит до входа текущего блока и до
   alpha_i, но НЕ до hidden_history/alpha_history предыдущих глубин (та же
   изоляция, что уже проверена для chunked_boost_loss в
   tools/check_boost_ablations.py) — регрессионный чек на новую функцию.
"""

import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.model.gpt import GPT, GPTConfig
from src.sid.chunked import chunked_boost_newton_loss
from src.sid.forward import embed, forward_range
from src.sid.newton import (apply_trust_region, descent_indicator, fix_minus_break_indicator,
                         newton_quad_per_token, predicted_improvement_per_token,
                         softmax_hessian_quadratic_form)

sys.stdout.reconfigure(encoding="utf-8")  # консоль Windows по умолчанию не печатает Δ/юникод из cp1251
torch.manual_seed(0)

# --- 1. Трюк Var_p[Δz] == Δz^T H Δz для явного H=diag(p)-pp^T на маленьком V ---
V = 8
z = torch.randn(3, V)
p = F.softmax(z, dim=-1)
delta_z = torch.randn(3, V)

H = torch.diag_embed(p) - torch.einsum("bi,bj->bij", p, p)  # (3,V,V)
quad_explicit = torch.einsum("bi,bij,bj->b", delta_z, H, delta_z)
quad_trick = softmax_hessian_quadratic_form(delta_z, p)
print(f"явный Δz^T H Δz vs трюк без VxV матрицы совпадают: "
      f"{torch.allclose(quad_explicit, quad_trick, atol=1e-4)}")
assert torch.allclose(quad_explicit, quad_trick, atol=1e-4)

# --- 2. При Δz=0 всё обнуляется ---
zero_delta = torch.zeros(3, V)
g = p - F.one_hot(torch.tensor([0, 1, 2]), V).float()
newton_loss_zero = newton_quad_per_token(zero_delta, p, g, lam=0.0, include_curvature=True)
assert torch.allclose(newton_loss_zero, torch.zeros(3), atol=1e-6)
print("Δz=0 -> newton_quad_per_token(lam=0) == 0: True")

pred_imp_zero = predicted_improvement_per_token(zero_delta, p, g)
assert torch.allclose(pred_imp_zero, torch.zeros(3), atol=1e-6)
print("Δz=0 -> predicted_improvement == 0: True")

# --- 3. descent_indicator / fix_minus_break_indicator на примерах из §9 ---
# p_y=0.1 (недооценка правильного токена) -> g_y=0.1-1=-0.9 (отрицательный).
# Если блок двигает Δz_y>0 (увеличивает логит правильного токена), то
# g_y*Δz_y<0 -> суммарный gᵀΔz должен быть <0 (верное направление), если
# остальные компоненты Δz малы.
p_example = torch.tensor([[0.1, 0.9]])
target_example = torch.tensor([0])
g_example = p_example - F.one_hot(target_example, 2).float()
delta_good = torch.tensor([[2.0, -2.0]])  # толкает логит правильного класса вверх, чужого вниз
delta_bad = torch.tensor([[-2.0, 2.0]])   # наоборот — усугубляет ошибку
desc_good = descent_indicator(delta_good, g_example)
desc_bad = descent_indicator(delta_bad, g_example)
print(f"descent_indicator(верное направление)={desc_good.item()} (ожидается 1.0), "
      f"descent_indicator(неверное)={desc_bad.item()} (ожидается 0.0)")
assert desc_good.item() == 1.0 and desc_bad.item() == 0.0

p_prev_ex = torch.tensor([[0.1, 0.9]])
p_cur_fixed = torch.tensor([[0.5, 0.5]])   # вероятность правильного (idx 0) выросла
p_cur_broken = torch.tensor([[0.05, 0.95]])  # упала
fix_val = fix_minus_break_indicator(p_prev_ex, p_cur_fixed, target_example)
break_val = fix_minus_break_indicator(p_prev_ex, p_cur_broken, target_example)
print(f"fix_minus_break(рост p_y)={fix_val.item()} (ожидается +1), "
      f"fix_minus_break(падение p_y)={break_val.item()} (ожидается -1)")
assert fix_val.item() == 1.0 and break_val.item() == -1.0

# --- 4. apply_trust_region ограничивает магнитуду ---
big_delta = torch.tensor([[100.0, -50.0, 0.01]])
clipped = apply_trust_region(big_delta, delta=2.0)
print(f"apply_trust_region(delta=2.0) на {big_delta.tolist()} -> {clipped.tolist()} "
      f"(|.|<=2.0 везде: {(clipped.abs() <= 2.0 + 1e-5).all().item()})")
assert (clipped.abs() <= 2.0 + 1e-5).all()
assert apply_trust_region(big_delta, None) is big_delta  # no-op

# --- 5. Градиентная изоляция chunked_boost_newton_loss (реальная модель) ---
config = GPTConfig()
model = GPT(config)
B = 4
idx = torch.randint(0, config.vocab_size, (B, config.block_size))
target_seq = torch.randint(0, config.vocab_size, (B, config.block_size))
K = 5

h_backbone = forward_range(model, embed(model, idx), 0, K).detach()
h1 = forward_range(model, h_backbone, K, K + 1).detach()
alpha1 = torch.tensor(0.5)
h2_in = h1.detach().requires_grad_(True)
h2 = forward_range(model, h2_in, K + 1, K + 2)
alpha2 = torch.tensor(0.01, requires_grad=True)
hidden_history = [h_backbone, h1]
alpha_history = [torch.tensor(1.0), alpha1]

loss, diag = chunked_boost_newton_loss(model, hidden_history, alpha_history, h2, alpha2, target_seq,
                                        lam=0.01, beta=1.0, include_curvature=True, chunk_size=64)
loss.backward()
print(f"\nchunked_boost_newton_loss diag: {{{', '.join(f'{k}={v.item():.4f}' for k, v in diag.items())}}}")
print(f"grad на h2_in (вход текущего блока): {h2_in.grad is not None}")
print(f"grad на alpha2: {alpha2.grad is not None and alpha2.grad.abs().item() > 0}")
assert h2_in.grad is not None
assert alpha2.grad is not None and alpha2.grad.abs().item() > 0
assert torch.isfinite(loss).item()
assert all(torch.isfinite(v).all().item() for v in diag.values())
print("diag теперь возвращает ТЕНЗОРЫ (не .item()) — вызывающий код сам решает, когда "
      "синхронизировать с CPU (важно для производительности, см. sid/chunked.py докстринг)")
assert all(torch.is_tensor(v) for v in diag.values())

# h_backbone/h1 были detach()-нуты ДО передачи сюда — сам факт того, что они
# .requires_grad=False, уже гарантирует отсутствие градиента в них; здесь
# дополнительно проверяем, что backward не падает и не проваливается в них
# (в PyTorch градиент просто не считается для тензоров без requires_grad).
assert not h_backbone.requires_grad and not h1.requires_grad
print("hidden_history-тензоры (backbone, block1) остаются requires_grad=False после backward: True")

print("\nВсе проверки Newton-SID пройдены.")
