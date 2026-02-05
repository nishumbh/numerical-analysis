import random
from helpers.matrix import Matrix


def make_diagonally_dominant_matrix(
    n: int,
    seed: int = 0,
    low: int = -5,
    high: int = 5,
    name: str = "A"
) -> Matrix:
    """
    Generates an n×n diagonally dominant matrix.
    Guaranteed to be nonsingular.
    """
    random.seed(seed)
    data = []

    for i in range(n):
        row = [random.randint(low, high) for _ in range(n)]
        row[i] = 0

        diag = sum(abs(x) for x in row) + random.randint(1, 5)
        row[i] = diag

        data.append(row)

    return Matrix(data, name=name)


def make_rhs(
    n: int,
    seed: int = 1,
    low: int = -10,
    high: int = 10,
    name: str = "b"
) -> Matrix:
    """
    Generates an n×1 RHS column vector.
    """
    random.seed(seed)
    return Matrix([[random.randint(low, high)] for _ in range(n)], name=name)


def make_random_matrix(
    n: int,
    seed: int = 0,
    low: int = -10,
    high: int = 10,
    name: str = "A"
) -> Matrix:
    """
    Generates a general dense n×n matrix.
    May require row exchanges (pivoting).
    """
    random.seed(seed)
    data = [
        [random.randint(low, high) for _ in range(n)]
        for _ in range(n)
    ]

    return Matrix(data, name=name)
