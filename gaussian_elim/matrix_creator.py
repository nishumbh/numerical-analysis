import random

def make_diagonally_dominant_matrix(n, seed=0, low=-5, high=5):
    random.seed(seed)
    A = []

    for i in range(n):
        row = [random.randint(low, high) for _ in range(n)]
        row[i] = 0

        diag = sum(abs(x) for x in row) + random.randint(1, 5)
        row[i] = diag

        A.append(row)

    return A

def make_rhs(n, seed=1, low=-10, high=10):
    random.seed(seed)
    return [random.randint(low, high) for _ in range(n)]
