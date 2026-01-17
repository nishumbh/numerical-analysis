"""
Secant Function for finding zeroes:

For a function f(x), the secant like through the points 
(x0, f(x0)) and (x1, f(x1)) has the equation:

              f(x1) - f(x0)
secant_line = -------------
                 x1 - x0

approximations:


                |  (x(k-1) - x(k))  |
x(k+1) = x(k) - |-------------------| . f(x(k))
                |f(x(k-1) - f(x(k)) |


Disadvantages: 
- Can fail at any stage (??? wtf lmao)

"""


from helpers import Equation, find_interval, Logger, find_interval_asym

class Secant:
    def __init__(self, eq: Equation):
        self.equation = eq

    def converge(self, x0, x1, tol=1e-8, max_iter=100):

        for _ in range(max_iter):
            fx0 = self.equation.solve(x0)
            fx1 = self.equation.solve(x1)

            if abs(fx1 - fx0) < 1e-14:
                raise ValueError("Division by near-zero in secant update")

            # if abs(fx1 - fx0) < 1e-14:
            p1 = (x1 - x0) / (fx0 - fx1)
            x2 = x1 - fx1 * p1

            if abs(x2 - x1) < tol:
                return x2

            x0,x1 = x1, x2

        raise RecursionError("Secant method did not converge")

if __name__ == '__main__':

    e1 = Equation(lambda x: x**2 - 4)
    secant = Secant(e1)
    a,b = find_interval(e1)
    print(a,b)
    root = secant.converge(1, 3)

    print(e1.solve(a))
    print(e1.solve(b))

    print(root)


