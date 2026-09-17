"""Проверка sid/cka.py::linear_cka: граничные случаи (совпадение, независимость,
инвариантность к линейному преобразованию — именно поэтому CKA~1 означает
"почти линейно связаны", а не "буквально идентичны") + градиент."""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.sid.cka import linear_cka

torch.manual_seed(0)
n, d = 512, 64

x = torch.randn(n, d)

# 1. CKA(X, X) должна быть ровно 1.
cka_self = linear_cka(x, x)
print(f"CKA(X, X) = {cka_self.item():.6f} (ожидается 1.0)")
assert abs(cka_self.item() - 1.0) < 1e-4

# 2. CKA инвариантна к линейному преобразованию (масштаб + сдвиг + поворот) —
# именно поэтому высокая CKA означает "блок делает что-то близкое к линейному
# преобразованию входа", а не "блок ничего не меняет".
rotation = torch.linalg.qr(torch.randn(d, d))[0]  # случайная ортогональная матрица
y_linear = (x @ rotation) * 3.7 + 1.5
cka_linear = linear_cka(x, y_linear)
print(f"CKA(X, поворот+масштаб+сдвиг X) = {cka_linear.item():.6f} (ожидается ~1.0)")
assert abs(cka_linear.item() - 1.0) < 1e-3

# 3. CKA(X, независимый шум) должна быть близка к 0.
y_indep = torch.randn(n, d)
cka_indep = linear_cka(x, y_indep)
print(f"CKA(X, независимый Y) = {cka_indep.item():.4f} (ожидается близко к 0)")
assert cka_indep.item() < 0.3

# 4. Промежуточный случай: Y = X + шум сопоставимой величины -> CKA где-то между.
y_mixed = x + torch.randn(n, d) * x.std()
cka_mixed = linear_cka(x, y_mixed)
print(f"CKA(X, X+шум) = {cka_mixed.item():.4f} (ожидается между 0 и 1, не крайности)")
assert 0.1 < cka_mixed.item() < 0.95

# 5. Градиент.
x_g = x.clone().requires_grad_()
y_g = y_mixed.clone().requires_grad_()
loss = linear_cka(x_g, y_g)
loss.backward()
print(f"градиент по x: max(abs)={x_g.grad.abs().max().item():.4f} (ожидается >0)")
print(f"градиент по y: max(abs)={y_g.grad.abs().max().item():.4f} (ожидается >0)")
assert x_g.grad.abs().max().item() > 0
assert y_g.grad.abs().max().item() > 0

print("\nВсе проверки пройдены.")
