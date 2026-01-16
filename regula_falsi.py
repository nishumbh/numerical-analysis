"""
Better way to solve for the bisection method

         f(b)−f(a)
c = a - --------
         f(a).(b−a)
"""

from helpers import Equation, find_interval, Logger

logger = Logger()

class RegulaFalsi:
    def __init__(self, eq):
        self.equation = eq

    def converge(self, a, b, tol=1e-6, max_iter=100):
        fa = self.equation.solve(a)
        fb = self.equation.solve(b)

        assert(fa * fb <= 0)

        if (fb - fa) == 0:
            return a

        for _ in range(max_iter):
            p1 = (fa * (b-a)) / (fb - fa)
            c = a - p1
            fc = self.equation.solve(c)

            logger.info(f"a = {a}, b = {b}, c = {c}, fc = {fc}")

            if abs(fc) < tol or abs(b - a) < tol:
                return c

            if fa * fc < 0:
                b = c
            else:
                a = c

if __name__ == '__main__':

    e1 = Equation(lambda x: x**5 + 2*x**3 + 45*x + 67)
    rf = RegulaFalsi(e1)
    a,b = find_interval(e1)
    root = rf.converge(a, b)

    print(e1.solve(a))
    print(e1.solve(b))

    print(root)
