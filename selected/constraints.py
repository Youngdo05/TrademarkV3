from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class ExposureRow:
    coeffs: tuple
    cap: float


@dataclass
class ExposureRows:
    n: int
    constraints: tuple
    _matrix_constraints: object = field(default=None, repr=False)
    _sparse: object = field(default=None, repr=False)
    _caps: object = field(default=None, repr=False)

    def __post_init__(self):
        if type(self.n) is not int or self.n <= 0 or not isinstance(self.constraints, tuple):
            raise ValueError("example geometry requires an asset count and immutable rows")
        for row in self.constraints:
            values = np.asarray(row.coeffs, dtype=float)
            if (not isinstance(row.coeffs, tuple) or values.shape != (self.n,)
                    or not np.isfinite(values).all() or (values < 0).any()
                    or not np.isfinite(row.cap) or row.cap < 0):
                raise ValueError("example exposure rows must be finite, nonnegative and aligned")

    def sparse_geometry(self):

        import numpy as np
        from scipy.sparse import csr_matrix
        if self._matrix_constraints is not self.constraints:
            data, indices, indptr = [], [], [0]
            for row in self.constraints:
                values = np.asarray(row.coeffs, dtype=float)
                columns = np.flatnonzero(values)
                data.extend(values[columns])
                indices.extend(columns)
                indptr.append(len(data))
            self._sparse = csr_matrix((data, indices, indptr), shape=(len(self.constraints), self.n))
            self._caps = np.asarray([row.cap for row in self.constraints], dtype=float)
            for array in (self._sparse.data, self._sparse.indices, self._sparse.indptr, self._caps):
                array.setflags(write=False)
            self._matrix_constraints = self.constraints
        return self._sparse, self._caps


def attach_exposure_limits(x, rows, indices=None):
    indices = np.arange(rows.n) if indices is None else np.asarray(indices)
    if (indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer)
            or len(indices) != len(set(indices.tolist()))
            or (indices < 0).any() or (indices >= rows.n).any()
            or x.shape != (len(indices),)):
        raise ValueError("example allocation indices must be unique and aligned")
    constraints = []
    matrix, caps = rows.sparse_geometry()
    restricted = matrix[:, indices].tocsr()
    active = np.flatnonzero(np.asarray(restricted.getnnz(axis=1)).reshape(-1))
    ceiling_vector = restricted[active] @ x <= caps[active] if len(active) else None
    if ceiling_vector is not None:
        constraints.append(ceiling_vector)
    ceiling_constraints = (active, ceiling_vector)
    return constraints, ceiling_constraints
