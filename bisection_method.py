# Bisection method to find zeroes for any f(n) = 0 where n is a real number.

from .helpers.base_poly import Equation

class Logger:
    @staticmethod
    def info(*msg):
        print(f"[INFO]: {msg}")

    @staticmethod
    def debug(msg, **kwargs):
        print(f"[DEBUG]: {msg} ; {kwargs=} ")

logger = Logger

class Bisection:
    def __init__(self, eq: Equation):
        self.equation = eq

    def find_interval(self, x0 = 0.0, step = 1.0, max_iter=100):
        # find [a, b] for which f(a) and f(b) is a range that passes 0
        f0 = self.equation.solve(x0)
        if f0 == 0:
            return x0, x0

        radius = step
        for _ in range(max_iter):
            a = x0 - radius
            b = x0 + radius
            fa = self.equation.solve(a)
            fb = self.equation.solve(b)

            if fa == 0:
                logger.info(a)
                return a, a
            if fb == 0:
                logger.info(b)
                return b, b
            if fa * fb < 0:
                logger.info(a, b)
                return a, b

            radius *= 2

        raise ValueError(f"could'nt find interval in {max_iter} iterations")

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
    a,b = bisect.find_interval()
    root = bisect.converge(a, b)

    print(e1.solve(a))
    print(e1.solve(b))

    print(root)
