"""Newton-SID — блок как локальный шаг Ньютона по CE в пространстве логитов.

Идея (обсуждение с пользователем 2026-09-21/22): вместо того чтобы блок
регрессировал явную Newton-цель Δz*=-(H+εI)⁻¹g (что требует деления на
p(1-p)+ε и может взрываться при уверенных неверных предсказаниях), блок
обучается через ТОЧНУЮ квадратичную Тейлор-аппроксимацию CE вокруг текущей
(detached) точки z_prev:

    CE(z_prev+Δz) ≈ CE(z_prev) + gᵀΔz + ½ΔzᵀHΔz

где для softmax-CE g=p-y и H=diag(p)-ppᵀ известны АНАЛИТИЧЕСКИ (не нужен
autograd для Hessian). Минимизация квадратичной формы по Δz даёт ровно
Δz*=-H⁻¹g — то есть сеть неявно учится делать Newton-шаг, просто спускаясь по
градиенту этой квадратичной формы, без единого деления на вероятность.

Ключевой трюк (чтобы не строить V×V матрицу H): для H=diag(p)-ppᵀ

    ΔzᵀHΔz = Σ_j p_j·Δz_j² - (Σ_j p_j·Δz_j)² = Var_{j~p}[Δz_j]

т.е. взвешенная (по текущим вероятностям) дисперсия компонент Δz — считается
за O(V) на токен вместо O(V²).

Softmax инвариантен к сдвигу (softmax(z)=softmax(z+c·1)) => H имеет нулевое
собственное направление (вдоль вектора из единиц) => H сингулярен. λ·Σ_jΔz_j²
(Tikhonov/Levenberg-Marquardt damping) регуляризует именно это направление —
это НЕ защита от деления на ноль (делений в этом модуле нет вообще), а
условие хорошей обусловленности квадратичной формы.
"""

import torch


def softmax_hessian_quadratic_form(delta_z, p):
    """ΔzᵀHΔz для H=diag(p)-ppᵀ, без материализации V×V матрицы.

    delta_z, p — любой одинаковый shape (..., V); возвращает (...,)."""
    mean_delta = (p * delta_z).sum(dim=-1)
    e_delta_sq = (p * delta_z.square()).sum(dim=-1)
    return e_delta_sq - mean_delta.square()


def _linear_and_curvature(delta_z, p_prev, g_prev, include_curvature):
    linear = (g_prev * delta_z).sum(dim=-1)
    if include_curvature:
        quad = softmax_hessian_quadratic_form(delta_z, p_prev)
    else:
        quad = torch.zeros_like(linear)
    return linear, quad


def predicted_improvement_per_token(delta_z, p_prev, g_prev, include_curvature=True):
    """Предсказанное (по квадратичной модели) изменение CE: ĝᵀΔz + ½ΔzᵀHΔz —
    БЕЗ damping-члена (damping — наша регуляризация обучения, а не часть
    реальной Taylor-аппроксимации CE). Используется только для диагностики
    (сравнение с фактическим ΔCE), не участвует в backward."""
    linear, quad = _linear_and_curvature(delta_z, p_prev, g_prev, include_curvature)
    return linear + 0.5 * quad


def newton_quad_per_token(delta_z, p_prev, g_prev, lam, include_curvature=True):
    """Обучающий вспомогательный лосс блока (§14/§19 обсуждения): линейный +
    curvature член Taylor-разложения CE + Tikhonov damping. include_curvature=
    False убирает curvature-член (вариант "Residual-SID", γ=0 в терминологии
    обсуждения — блок видит только первый порядок, без кривизны)."""
    linear, quad = _linear_and_curvature(delta_z, p_prev, g_prev, include_curvature)
    damping = lam * delta_z.square().sum(dim=-1)
    return linear + 0.5 * quad + 0.5 * damping


def apply_trust_region(delta_z, delta):
    """Ограничивает |Δz_j|<=delta поэлементно через delta·tanh(Δz/delta) —
    квадратичная Taylor-аппроксимация верна только локально (большой Δz может
    увести далеко от точки разложения), tanh мягко "подрезает" магнитуду, не
    убивая градиент (в отличие от жёсткого clip). delta=None -> no-op (без
    trust region, "естественный" вариант)."""
    if delta is None:
        return delta_z
    return delta * torch.tanh(delta_z / delta)


def descent_indicator(delta_z, g_prev):
    """1.0 для токенов, где блок двигается в сторону, УМЕНЬШАЮЩУЮ CE в первом
    порядке (gᵀΔz<0 — корректное direction), иначе 0.0. Доля таких токенов —
    диагностика §22 обсуждения (P(gᵀΔz<0)), должна быть заметно >0.5, если
    блок действительно направленно исправляет ошибку, а не шумит."""
    dot = (g_prev * delta_z).sum(dim=-1)
    return (dot < 0).to(delta_z.dtype)


def fix_minus_break_indicator(p_prev, p_cur, target):
    """+1 если вероятность истинного токена ВЫРОСЛА после коррекции блока
    ("Fix"), -1 если УПАЛА ("Break"), 0 без изменений. Среднее по токенам —
    операционализация §25 обсуждения: пользователь явно указал, что CKA≈1
    между соседними блоками сам по себе НЕ означает патологический коллапс,
    если Fix-Break остаётся заметно положительным (блок всё ещё полезен,
    просто линейно/просто это делает) — а вот Fix-Break→0 при ЛЮБОМ CKA
    означает, что блок перестал приносить пользу."""
    p_prev_true = p_prev.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    p_cur_true = p_cur.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    return torch.sign(p_cur_true - p_prev_true)
