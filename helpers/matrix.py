import math
from typing import Optional

import numpy as np

THRES = 1e-10


# helpers/matrix.py
class Matrix:
    """
    Reduction convention (sum, mean, var, std, argmax):
        axis=None -> a plain Python scalar (argmax: flat row-major index)
        axis=0    -> 1 x n Matrix  (one value per column)
        axis=1    -> m x 1 Matrix  (one value per row)
    """

    def __init__(self, data: list = None, name="A", as_row: bool = False):
        """
        2D input  : [[1, 2], [3, 4]]  -> used as-is (rows are copied)
        1D input  : [1, 2, 3]         -> n x 1 column by default,
                                         1 x n row if as_row=True
        as_row is ignored for 2D input.
        """
        if data is None:
            raise ValueError("Matrix cannot be empty")
        data = list(data)  # also accepts tuples, NumPy arrays, other Matrices
        if not data:
            raise ValueError("Matrix cannot be empty")

        is_flat = all(not hasattr(r, "__len__") for r in data)

        if is_flat:
            self.matrix = [data] if as_row else [[v] for v in data]
        else:
            # Copy the rows so we never alias the caller's lists
            self.matrix = [list(r) if hasattr(r, "__len__") else r for r in data]

        self.name = name
        self._validate()

    # -------------------------------

    @property
    def rows(self):
        return len(self.matrix)

    @property
    def cols(self):
        return len(self.matrix[0])

    @property
    def shape(self):
        return (self.rows, self.cols)

    @property
    def is_square(self):
        return self.rows == self.cols

    @property
    def T(self):
        return self.transpose()

    # -------------------------------

    def _validate(self):
        if not self.matrix:
            raise ValueError("Matrix cannot be empty")

        if not hasattr(self.matrix[0], "__len__"):
            raise ValueError("Cannot mix scalars and rows in one matrix")

        width = len(self.matrix[0])

        if width == 0:
            raise ValueError("Matrix cannot have zero columns")

        for row in self.matrix:
            if not hasattr(row, "__len__"):
                raise ValueError("Cannot mix scalars and rows in one matrix")
            if len(row) != width:
                raise ValueError("All rows must have the same length")

    def _check_same_shape(self, other):
        if not isinstance(other, Matrix):
            raise TypeError(f"Expected a Matrix, got {type(other).__name__}")
        if self.shape != other.shape:
            raise ArithmeticError(f"Shape mismatch, {self.shape}, {other.shape}")

    @staticmethod
    def _check_axis(axis):
        if axis not in (None, 0, 1):
            raise ValueError("axis must be 0, 1, or None")

    def transpose(self, inplace=False):
        t = Matrix.from_shape(self.cols, self.rows)
        for i in range(self.rows):
            for j in range(self.cols):
                t[j, i] = self[i, j]

        if inplace:
            self.matrix = t.matrix
            return self

        return t

    # -------------------------------

    @staticmethod
    def from_shape(rows, cols):
        if rows <= 0 or cols <= 0:
            raise ValueError("rows and cols must be positive")
        return Matrix([[0] * cols for _ in range(rows)])

    @classmethod
    def from_matrix(cls, other):
        return cls.from_shape(*other.shape)

    @classmethod
    def from_array(cls, data, cols):
        """Build a matrix from a flat list, `cols` elements per row."""
        data = list(data)
        if cols <= 0 or len(data) % cols != 0:
            raise ValueError(f"Cannot split {len(data)} elements into rows of {cols}")
        # (original used data[step:step+step], which is a bug)
        return Matrix([data[i:i + cols] for i in range(0, len(data), cols)])

    @classmethod
    def inverse(cls, other):
        if not other.is_square:
            raise ArithmeticError("Not a square matrix")

        n = other.rows

        # ---- GAUSSIAN ELIMINATION ----
        U, L, P = ga(other)

        # ---- RESULT ----
        inverse = cls.from_shape(n, n)
        inverse.name = f"{other.name}^(-1)"

        # ---- SOLVE Ax = e_i FOR EACH COLUMN ----
        for j in range(n):
            e = cls.from_shape(n, 1)

            # Create the j-th column of the identity matrix
            e[j, 0] = 1

            # Solve Ax = e
            y = forward_sub(L=L, P=P, b=e)
            x = back_sub(U=U, y=y)

            # x becomes the j-th column of A^-1
            inverse.set_col(j, x)

        return inverse

    @staticmethod
    def identity(n):
        m = Matrix.from_shape(n, n)
        for i in range(n):
            m[i, i] = 1
        return m

    @staticmethod
    def vstack(*matrices):
        """Stack matrices vertically (they must share the same column count)."""
        if not matrices:
            raise ValueError("vstack needs at least one matrix")
        cols = matrices[0].cols
        for m in matrices:
            if m.cols != cols:
                raise ArithmeticError(
                    f"vstack column mismatch: {cols} vs {m.cols}"
                )
        return Matrix([row[:] for m in matrices for row in m.matrix])

    @staticmethod
    def hstack(*matrices):
        """Stack matrices horizontally (they must share the same row count)."""
        if not matrices:
            raise ValueError("hstack needs at least one matrix")
        rows = matrices[0].rows
        for m in matrices:
            if m.rows != rows:
                raise ArithmeticError(
                    f"hstack row mismatch: {rows} vs {m.rows}"
                )
        return Matrix([
            [v for m in matrices for v in m.matrix[i]]
            for i in range(rows)
        ])

    def to_list(self):
        return self.matrix

    # -------------------------------

    def copy(self):
        return Matrix([row[:] for row in self.matrix], name=self.name)

    # -------------------------------

    def __getitem__(self, key):
        if isinstance(key, slice):
            return Matrix(self.matrix[key])

        if isinstance(key, tuple):
            row, col = key
            return float(self.matrix[row][col])

        return self.matrix[key]

    def __setitem__(self, key, value):
        if isinstance(key, int):
            self.matrix[key] = value
            self._validate()
            return

        if isinstance(key, tuple):
            row, col = key
            self.matrix[row][col] = value
            return

        if isinstance(key, slice):
            if len(value) != len(range(*key.indices(self.rows))):
                raise ValueError("Slice assignment size mismatch")
            self.matrix[key] = value
            self._validate()
            return

        raise TypeError("Invalid index type")

    # -------------------------------

    def __iter__(self):
        return iter(self.matrix)

    def __repr__(self):
        return str(self.matrix)

    def __str__(self):
        return str(self.matrix)

    def __eq__(self, other):
        return isinstance(other, Matrix) and self.matrix == other.matrix

    # -------------------------------
    # Row / column access
    # -------------------------------

    def row(self, i):
        """Return row i as a 1 x n Matrix."""
        if not -self.rows <= i < self.rows:
            raise IndexError(f"Row index {i} out of range for {self.rows} rows")
        return Matrix([self.matrix[i][:]])

    def col(self, j):
        """Return column j as an m x 1 Matrix."""
        if not -self.cols <= j < self.cols:
            raise IndexError(f"Column index {j} out of range for {self.cols} cols")
        return Matrix([[row[j]] for row in self.matrix])

    def col_list(self, j):
        """Return column j as a plain Python list (the old col() behaviour)."""
        if not -self.cols <= j < self.cols:
            raise IndexError(f"Column index {j} out of range for {self.cols} cols")
        return [row[j] for row in self.matrix]

    def set_col(self, j, values):
        """`values` may be a list or an m x 1 Matrix."""
        if isinstance(values, Matrix):
            if values.shape != (self.rows, 1):
                raise ValueError("Column size mismatch")
            values = values.flatten()
        if len(values) != self.rows:
            raise ValueError("Column size mismatch")
        for i in range(self.rows):
            self.matrix[i][j] = values[i]

    # -------------------------------
    # Functional helpers
    # -------------------------------

    def map(self, fn):
        if not callable(fn):
            raise Exception(f"{fn} not callable, it is type: {type(fn)}")
        return Matrix([[fn(x) for x in row] for row in self.matrix])

    def reduce(self, fn):
        if not callable(fn):
            raise Exception(f"{fn} not callable, it is type: {type(fn)}")
        return fn(self.matrix)

    def clip(self, min_val=None, max_val=None):
        """Bound every element to [min_val, max_val]; either bound may be None."""
        if min_val is not None and max_val is not None and min_val > max_val:
            raise ValueError("min_val cannot be greater than max_val")

        def _clip(x):
            if min_val is not None and x < min_val:
                return min_val
            if max_val is not None and x > max_val:
                return max_val
            return x

        return self.map(_clip)

    # -------------------------------
    # Reductions
    # -------------------------------

    def sum(self, axis: Optional[int] = None):
        """
        return a sum of the matrix,
            None: full sum of the matrix
                ret: num
            0: Sum of each col
                ret: Matrix 1xn
            1: Sum of each row
                ret: Matrix nx1
        """
        self._check_axis(axis)

        if axis is None:
            return sum(sum(row) for row in self.matrix)

        if axis == 0:
            return Matrix([[
                sum(row[col] for row in self.matrix)
                for col in range(self.cols)
            ]])

        return Matrix([[sum(row)] for row in self.matrix])

    def mean(self, axis: Optional[int] = None):
        """Same axis convention as sum()."""
        self._check_axis(axis)

        if axis is None:
            return self.sum() / (self.rows * self.cols)
        if axis == 0:
            return self.sum(0) / self.rows
        return self.sum(1) / self.cols

    def var(self, axis: Optional[int] = None, ddof: int = 0):
        """
        Variance with the same axis convention as sum().
        ddof=0 is the population variance, ddof=1 the sample variance.
        """
        self._check_axis(axis)

        n = {None: self.rows * self.cols, 0: self.rows, 1: self.cols}[axis]
        if n - ddof <= 0:
            raise ValueError("Not enough elements for the requested ddof")

        if axis is None:
            mu = self.mean()
            return sum((v - mu) ** 2 for row in self.matrix for v in row) / (n - ddof)

        mu = self.mean(axis)
        if axis == 0:
            return Matrix([[
                sum((self.matrix[i][j] - mu[0, j]) ** 2 for i in range(self.rows))
                / (n - ddof)
                for j in range(self.cols)
            ]])

        return Matrix([[
            sum((v - mu[i, 0]) ** 2 for v in self.matrix[i]) / (n - ddof)
        ] for i in range(self.rows)])

    def std(self, axis: Optional[int] = None, ddof: int = 0):
        v = self.var(axis, ddof)
        if axis is None:
            return math.sqrt(v)
        return v.map(math.sqrt)

    def argmax(self, axis: Optional[int] = None):
        """
        None: flat (row-major) index of the largest element -> int
        0:    row index of the max in each column           -> Matrix 1xn
        1:    column index of the max in each row           -> Matrix mx1
        Ties resolve to the first occurrence.
        """
        self._check_axis(axis)

        if axis is None:
            flat = self.flatten()
            return max(range(len(flat)), key=lambda k: flat[k])

        if axis == 0:
            return Matrix([[
                max(range(self.rows), key=lambda i: self.matrix[i][j])
                for j in range(self.cols)
            ]])

        return Matrix([[
            max(range(self.cols), key=lambda j: row[j])
        ] for row in self.matrix])

    def norm(self, ord=2):
        """
        Treats the matrix as a flattened vector:
            ord=2      -> Euclidean / Frobenius norm
            ord=1      -> sum of absolute values
            ord=np.inf -> largest absolute value
        """
        values = self.flatten()
        if ord == 2:
            return math.sqrt(sum(v * v for v in values))
        if ord == 1:
            return sum(abs(v) for v in values)
        if ord == float("inf"):
            return max(abs(v) for v in values)
        raise ValueError("ord must be 1, 2, or inf")

    # -------------------------------
    # Shape manipulation
    # -------------------------------

    def flatten(self):
        flat_arr = []
        for row in self.matrix:
            for col in row:
                flat_arr.append(col)

        return flat_arr

    def reshape(self, rows, cols):
        """Row-major reshape. One of rows/cols may be -1 to be inferred."""
        total = self.rows * self.cols

        if rows == -1 and cols == -1:
            raise ValueError("Only one dimension can be -1")
        if rows == -1:
            if cols <= 0 or total % cols != 0:
                raise ValueError(f"Cannot reshape {self.shape} into (?, {cols})")
            rows = total // cols
        elif cols == -1:
            if rows <= 0 or total % rows != 0:
                raise ValueError(f"Cannot reshape {self.shape} into ({rows}, ?)")
            cols = total // rows

        if rows <= 0 or cols <= 0 or rows * cols != total:
            raise ValueError(f"Cannot reshape {self.shape} into ({rows}, {cols})")

        return Matrix.from_array(self.flatten(), cols)

    # -------------------------------
    # Comparison
    # -------------------------------

    def allclose(self, other, tol=1e-9):
        """True if shapes match and every |a - b| <= tol."""
        if not isinstance(other, Matrix) or self.shape != other.shape:
            return False
        return all(
            abs(a - b) <= tol
            for ra, rb in zip(self.matrix, other.matrix)
            for a, b in zip(ra, rb)
        )

    # -------------------------------
    # Arithmetic
    # -------------------------------

    def __add__(self, other):
        if isinstance(other, (int, float, np.number)):
            return Matrix([
                [self[i, j] + other for j in range(self.cols)]
                for i in range(self.rows)
            ])

        self._check_same_shape(other)
        return Matrix([
            [self[i, j] + other[i, j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        if isinstance(other, (int, float, np.number)):
            return Matrix([
                [self[i, j] - other for j in range(self.cols)]
                for i in range(self.rows)
            ])

        self._check_same_shape(other)
        return Matrix([
            [self[i, j] - other[i, j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def __rsub__(self, other):
        return (-self) + other

    def __neg__(self):
        return self.map(lambda v: -v)

    def __matmul__(self, other):
        if not isinstance(other, Matrix):
            raise TypeError(f"Expected a Matrix, got {type(other).__name__}")
        if self.cols != other.rows:
            raise ArithmeticError(f"Invalid dimensions for multiplication, {self.shape}, {other.shape}")

        result = Matrix.from_shape(self.rows, other.cols)

        for i in range(self.rows):
            for j in range(other.cols):
                result[i, j] = sum(
                    self[i, k] * other[k, j]
                    for k in range(self.cols)
                )
        return result

    def multiply_elementwise(self, other):
        """Hadamard product: same shape required."""
        self._check_same_shape(other)
        return Matrix([
            [a * b for a, b in zip(ra, rb)]
            for ra, rb in zip(self.matrix, other.matrix)
        ])

    def __mul__(self, other):
        # scalar multiplication only
        if isinstance(other, (int, float, np.number)):
            return Matrix(
                [[other * v for v in row] for row in self.matrix],
                name=self.name
            )
        raise TypeError("Use @ for matrix multiplication or multiply_elementwise()")

    def __truediv__(self, other):
        # scalar division only
        if isinstance(other, (int, float, np.number)):
            return Matrix(
                [[v / other for v in row] for row in self.matrix],
                name=self.name
            )
        raise TypeError("Cant Divide 2 Matrices")

    def __rmul__(self, other):
        return self * other

    def __pow__(self, exponent):
        """Element-wise power (NOT the matrix power)."""
        if not isinstance(exponent, (int, float, np.number)):
            raise TypeError("Exponent must be a scalar; __pow__ is element-wise")
        return self.map(lambda v: v ** exponent)

    def __abs__(self):
        return Matrix([
            [abs(v) for v in row]
            for row in self.matrix
        ])

    # -------------------------------


# gaussian_elim/ga_with_row_exchanges.py
def ga(A: Matrix):
    A = A.copy()
    n = A.rows

    L = Matrix.identity(n)
    U = A
    P = Matrix.identity(n)
    P.name = "P"
    U.name = "U"
    L.name = "L"

    for k in range(n):  # ---- PARTIAL PIVOTING ----
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


def forward_sub(*, L: Matrix, P: Matrix, b: Matrix):
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
