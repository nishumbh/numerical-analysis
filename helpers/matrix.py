class Matrix:
    def __init__(self, data: list, name="A"):
        self.matrix = data
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

    # -------------------------------

    def _validate(self):
        if not self.matrix:
            raise ValueError("Matrix cannot be empty")

        width = len(self.matrix[0])
        for row in self.matrix:
            if len(row) != width:
                raise ValueError("All rows must have the same length")

    def _check_same_shape(self, other):
        if self.shape != other.shape:
            raise ArithmeticError("Shape mismatch")

    # -------------------------------

    @staticmethod
    def from_shape(rows, cols):
        return Matrix([[0] * cols for _ in range(rows)])

    @classmethod
    def from_matrix(cls, other):
        return cls.from_shape(*other.shape)


    @staticmethod
    def identity(n):
        m = Matrix.from_shape(n, n)
        for i in range(n):
            m[i, i] = 1
        return m

    # -------------------------------

    def copy(self):
        return Matrix([row[:] for row in self.matrix])

    # -------------------------------

    def __getitem__(self, key):
        if isinstance(key, slice):
            return Matrix(self.matrix[key])

        if isinstance(key, tuple):
            row, col = key
            return self.matrix[row][col]

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
        return f"Matrix({self.matrix})"

    def __str__(self):
        return f"""{self.name} = [\n\t{"\n\t".join(" ".join(f"{v:6.2f}" for v in row) for row in self.matrix)}\n]"""
    def __eq__(self, other):
        return isinstance(other, Matrix) and self.matrix == other.matrix

    # -------------------------------

    def col(self, j):
        return [row[j] for row in self.matrix]

    def set_col(self, j, values):
        if len(values) != self.rows:
            raise ValueError("Column size mismatch")
        for i in range(self.rows):
            self.matrix[i][j] = values[i]

    def map(self, fn):
        return Matrix([[fn(x) for x in row] for row in self.matrix])

    # -------------------------------

    def __add__(self, other):
        self._check_same_shape(other)
        return Matrix([
            [self[i, j] + other[i, j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def __sub__(self, other):
        self._check_same_shape(other)
        return Matrix([
            [self[i, j] - other[i, j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def __matmul__(self, other):
        if self.cols != other.rows:
            raise ArithmeticError("Invalid dimensions for multiplication")

        result = Matrix.from_shape(self.rows, other.cols)

        for i in range(self.rows):
            for j in range(other.cols):
                result[i, j] = sum(
                    self[i, k] * other[k, j]
                    for k in range(self.cols)
                )
        return result

    def __mul__(self, other):
        # scalar multiplication only
        if isinstance(other, (int, float)):
            return Matrix(
                [[other * v for v in row] for row in self.matrix],
                name=self.name
            )
        raise TypeError("Use @ for matrix multiplication")

    def __rmul__(self, other):
        return self * other

    def __abs__(self):
        return Matrix([
            [abs(v) for v in row]
            for row in self.matrix
        ])

    # -------------------------------

    def T(self, inplace=True):
        t = Matrix.from_shape(self.cols, self.rows)
        for i in range(self.rows):
            for j in range(self.cols):
                t[j, i] = self.matrix[i][j]

        if inplace:
            self.matrix = t.matrix
            return

        return t
