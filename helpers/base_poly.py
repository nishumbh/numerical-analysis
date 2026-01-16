
class Equation:
    def __init__(self, func) -> None:
        self.func = func

    def solve(self, x):
        return self.func(x)

    def update(self, func):
        self.func = func

