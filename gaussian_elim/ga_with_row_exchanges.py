from gaussian_elim.matrix_creator import (
    make_diagonally_dominant_matrix,
    make_random_matrix,
    make_rhs,
)
from helpers.matrix import Matrix

THRES = 1e-10

def forward_sub(*, L: Matrix, P:Matrix, b: Matrix):
    Pb = P @ b
    n = L.rows
    y = Matrix.from_shape(n, 1)
    y.name = "y"

    for i in range(n):
        y[i, 0] = Pb[i, 0] - sum(L[i, j] * y[j, 0] for j in range(i))

    return y


def back_sub(*, U: Matrix, y: Matrix):
    n = U.rows
    x = Matrix.from_shape(n, 1)
    x.name = "x"

    for i in range(n - 1, -1, -1):
        x[i, 0] = (
            y[i, 0]
            - sum(U[i, j] * x[j, 0] for j in range(i + 1, n))
        ) / U[i, i]

    return x


# ----------------------------------------
# Gaussian Elimination with Partial Pivoting
# PA = LU
# ----------------------------------------
def ga(A: Matrix):
    A = A.copy()
    n = A.rows

    L = Matrix.identity(n)
    U = A
    P = Matrix.identity(n)
    P.name = "P"
    U.name = "U"
    L.name = "L"

    for k in range(n):
        # ---- PARTIAL PIVOTING ----
        pivot_row = max(
            range(k, n),
            key=lambda i: abs(U[i, k])
        )

        if abs(U[pivot_row, k]) < THRES:
            raise ZeroDivisionError("Matrix is singular or nearly singular")

        # ---- ROW SWAPS ----
        if pivot_row != k:
            U[k], U[pivot_row] = U[pivot_row], U[k]
            P[k], P[pivot_row] = P[pivot_row], P[k]

            for j in range(k):
                L[k][j], L[pivot_row][j] = L[pivot_row][j], L[k][j]

        # ---- ELIMINATION ----
        for i in range(k + 1, n):
            factor = U[i, k] / U[k, k]
            L[i][k] = factor

            for j in range(k, n):
                U[i, j] -= factor * U[k, j]
                if abs(U[i, j]) < THRES:
                    U[i, j] = 0.0

    return U, L, P

def validate_ga(*, A: Matrix, b: Matrix, U: Matrix, L: Matrix, P: Matrix):

    # ---- Reconstruction check: PA ≈ LU ----
    PA = P @ A0
    LU = L @ U

    max_err = max(
        abs(PA[i, j] - LU[i, j])
        for i in range(n)
        for j in range(n)
    )
    print("reconstruction error (PA - LU):", max_err)


def residuals(*, x, A, b):

    # ---- Residual check: Ax - b ----
    r = A @ x
    residual = max(
        abs(r[i, 0] - b[i, 0])
        for i in range(A.rows)
    )
    print("residual ||Ax - b||∞:", residual)

if __name__ == "__main__":
    n = 5

    A0 = make_random_matrix(n, seed=42, name="A")
    b0 = make_rhs(n, seed=123, name="b")

    U, L, P = ga(A0)

    # ---- Solve Ax = b ----
    y = forward_sub(
        L=L,
        P=P,
        b=b0
    )
    x = back_sub(
        U=U,
        y=y
    )

    validate_ga(
        A=A0,
        b=b0,
        U=U,
        L=L,
        P=P
    )

    residuals(x= x, A=A0, b=b0)

    print("\n--- Matrices ---")
    print(A0)
    print(P)
    print(L)
    print(U)
    print(x)
