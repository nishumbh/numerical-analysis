# Bisection method to find zeroes for any f(n) = 0 where n is a real number.

from helpers import Equation, find_interval, Logger


logger = Logger

class Bisection:
    def __init__(self, eq: Equation):
        self.equation = eq


    def converge(self, a, b, tol=1e-6, max_iter=100):
        fa = self.equation.solve(a)
        fb = self.equation.solve(b)

        assert(fa * fb <= 0)

        for _ in range(max_iter):
            m = (a + b) / 2
            fm = self.equation.solve(m)

            logger.info(f"a = {a}, b = {b}, c = {m}, fc = {fm}")

            if abs(fm) < tol or abs(b - a) < tol:
                return m

            if fa * fm < 0:
                b = m
            else:
                a = m

        return (a + b) / 2


if __name__ == '__main__':

    e1 = Equation(lambda x: 20*x**5 + 2*x**3 + 45*x + 67)
    bisect = Bisection(e1)
    a,b = find_interval(e1)
    root = bisect.converge(a, b)

    print(e1.solve(a))
    print(e1.solve(b))

    print(root)
