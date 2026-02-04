from time import time
from matrix_creator import make_diagonally_dominant_matrix, make_rhs

THRES = 1e-12

"""
a = \
[
    [ 2,  1, 1],
    [ 4, -6, 0],
    [-2,  7, 2],
]

# Operations:
# E21
# E31
# E32

Algorithm:
Make it Upper Triangular Matrix

A = [
        [ 2,  1, 1],
        [ X, -6, 0],
        [ X,  X, 2],
    ]

"""

def matrix_mult(A, B):
    if len(A[0]) != len(B):
        raise Exception("Rows and Columns of Matrices are not equal")

    C = [[0] * len(B[0]) for _ in range(len(A))]
    for row in range(len(A)):
        for col in range(len(B[0])):
            for elt in range(len(B)):
                intermediate = A[row][elt] * B[elt][col]
                C[row][col] += intermediate if abs(intermediate) > THRES else 0

    return C

def printm(x, name="A"):
    # Nobody asks about this
    print(f"""{name} = [\n\t{"\n\t".join(" ".join(f"{v:6.2f}" if v!=0 else " "*6 for v in row) if isinstance(row, list) else str(round(row, 2)) for row in x)}\n]""")

def forward_sub(L, b):
    n = len(L)
    y = [0.0]*n
    for i in range(n):
        y[i] = b[i] - sum(L[i][j]*y[j] for j in range(i))
    return y

def back_sub(U, y):
    n = len(U)
    x = [0.0]*n
    for i in range(n-1, -1, -1):
        x[i] = (y[i] - sum(U[i][j]*x[j] for j in range(i+1, n))) / U[i][i]
    return x

def matvec(A, x):
    return [sum(A[i][j]*x[j] for j in range(len(x))) for i in range(len(A))]

# Iteration 1
# Only for MxM matrices
def ga(A, b=None):
    n = len(A)
    L = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

    for rowIdx in range(n):
        for colIdx in range(rowIdx):  # only below diagonal

            curr_cell = A[rowIdx][colIdx]
            if abs(curr_cell) < THRES:
                A[rowIdx][colIdx] = 0.0
                continue

            pivot = rowIdx - 1
            while pivot >= 0 and abs(A[pivot][colIdx]) < THRES:
                pivot -= 1

            if pivot < 0:
                raise ZeroDivisionError("No valid pivot found")

            fac_cell = A[pivot][colIdx]
            factor = -curr_cell / fac_cell

            for j in range(colIdx, n):
                A[rowIdx][j] += factor * A[pivot][j]
                if abs(A[rowIdx][j]) < THRES:
                    A[rowIdx][j] = 0.0

            if b is not None:
                b[rowIdx] += factor * b[pivot]
                if abs(b[rowIdx]) < THRES:
                    b[rowIdx] = 0.0

            L[rowIdx][pivot] = -factor

    return A, b, L

n = 50
A0 = make_diagonally_dominant_matrix(n, 69)
b0 = make_rhs(n, 123)

U, fb, L = ga([row[:] for row in A0], b0[:])

A_recon = matrix_mult(L, U)

y = forward_sub(L, fb)
x = back_sub(U, y)

max_err = max(
    abs(A0[i][j] - A_recon[i][j])
    for i in range(n)
    for j in range(n)
)
print("reconstruction error:", max_err)

r = matvec(A0, x)
residual = max(abs(r[i] - b0[i]) for i in range(n))
print("residual:", residual)
