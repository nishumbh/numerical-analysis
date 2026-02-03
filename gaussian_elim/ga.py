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
                C[row][col] += intermediate if intermediate > THRES else 0

    return C

def printm(x, name="A"):
    # Nobody asks about this
    print(f"""{name} = [\n\t{"\n\t".join(" ".join(f"{v:6.2f}" if v!=0 else " "*6 for v in row) if isinstance(row, list) else str(round(row, 2)) for row in x)}\n]""")



def mult_with_factor(A, factor, rowIdx, colIdx):
    n = len(A)
    # Build elementary matrix for L
    dI = [
        [1.0 if cx == rx else factor if cx == colIdx and rx == rowIdx else 0.0 for cx in range(n)]
        for rx in range(n)
    ]
    
    # Instead of full matrix multiply, apply factor in-place
    pivot_row = rowIdx - 1
    while pivot_row >= 0 and abs(A[pivot_row][colIdx]) < THRES:
        pivot_row -= 1

    for j in range(len(A[rowIdx])):
        A[rowIdx][j] += factor * A[pivot_row][j]
        # Optional: zero-out very small numbers
        if abs(A[rowIdx][j]) < THRES:
            A[rowIdx][j] = 0.0

    return A, dI

def mult_b_with_factor(b, factor, row_to, row_from):
    b[row_to] += factor * b[row_from]
    if abs(b[row_to]) < THRES:
        b[row_to] = 0.0

# Iteration 1
# Only for MxM matrices
def ga(A, b=None):
    dE_A = {}
    for rowIdx in range(len(A)):
        for colIdx in range(len(A[rowIdx])):
            if rowIdx == colIdx:
                break

            # Current cell to make 0 = A[rowIdx][colIdx]
            # Factor to multiply the cell by = A[rowIdx-1][colIdx]

            curr_cell = A[rowIdx][colIdx]
            if curr_cell == 0:
                continue

            fac_cell = 0
            cnt = 1
            while fac_cell == 0: fac_cell = A[rowIdx-cnt][colIdx]; cnt += 1

            factor = -1 * (curr_cell/fac_cell)
            A, dE_A[f"{rowIdx}{colIdx}"] = mult_with_factor(**{
                "A": A,
                "factor": factor,
                "rowIdx": rowIdx,
                "colIdx": colIdx
            })

            mult_b_with_factor(b, factor, rowIdx, rowIdx-cnt+1)

        #     printm(b)
        #     printm(A, f"A{rowIdx}{colIdx}")
        #     printm(b, f"b{rowIdx}{colIdx}")
        #
        #
        # print()

    # print("Eliminations:")
    # [printm(val, key) for key, val in dE_A.items()]

    aL = list(dE_A.values())
    L = aL[0]
    for dL in aL[1:]:
        L = matrix_mult(L, dL)

    printm(L, "L")
    return A, b

n = 50
a = make_diagonally_dominant_matrix(n, 69)
b = make_rhs(n, int(time()))
U, fb = ga(a, b)

print("\n\n\n")
printm(U, "U")
printm(fb, "fb")


