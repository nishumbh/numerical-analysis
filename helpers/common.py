from operator import eq
from sys import maxsize


def find_interval(equation, x0 = 0.0, step = 1.0, max_iter=100):
    # find [a, b] for which f(a) and f(b) is a range that passes 0
    f0 = equation.solve(x0)
    if f0 == 0:
        return x0, x0

    radius = step
    for _ in range(max_iter):
        a = x0 - radius
        b = x0 + radius
        fa = equation.solve(a)
        fb = equation.solve(b)

        if fa == 0:
            return a, a
        if fb == 0:
            return b, b
        if fa * fb < 0:
            return a, b

        radius += step

    raise ValueError(f"could'nt find interval in {max_iter} iterations")


def find_interval_asym(equation, x0=0.0, step=1.0, max_iter=100, tol=1e-6, refine=5):

    f0 = equation.solve(x0)

    if abs(f0) < tol:
        return x0,x0

    dir = -1 if f0 > 0 else 1

    x_prev = x0
    f_prev = f0
    radius = step

    for _ in range(max_iter):
        x_curr = x0 + dir * radius
        f_curr = equation.solve(x_curr)

        if abs(f_curr) < tol:
            return x_curr, x_curr

        if f_prev * f_curr < 0:
            a , b = sorted((x_prev, x_curr))
            break

        x_prev, f_prev = x_curr, f_curr

        radius *= 2
    else:
        radius = step
        for _ in range(max_iter):
            a = x0 - radius
            b = x0 + radius
            fa = equation.solve(a)
            fb = equation.solve(b)

            if abs(fa) < tol:
                return a, a
            if abs(fb) < tol:
                return b, b

            if fa * fb < 0:
                break

            radius *= 2
        else:
            raise ValueError("Could not bracket root")

    for _ in range(refine):
        m = 0.5 * (a + b)
        fm = equation.solve(m)

        if abs(fm) < tol:
            return m, m

        if equation.solve(a) * fm < 0:
            b = m
        else:
            a = m

    return a, b
