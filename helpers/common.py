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

        radius *= 2

    raise ValueError(f"could'nt find interval in {max_iter} iterations")
