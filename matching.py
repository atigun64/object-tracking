class Matrix:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.data = [[0 for _ in range(cols)] for _ in range(rows)]

    def set(self, row, col, value):
        self.data[row][col] = value

    def get(self, row, col):
        return self.data[row][col]


def hungarian_match(matrix, maximize=False):
    """
    Compute an optimal assignment for the given cost/score matrix using the
    Hungarian algorithm.

    Args:
        matrix: either a Matrix instance (from this module) or a 2D list of
            numeric values. Rows represent workers (left side), columns
            represent tasks (right side).
        maximize: if True, treats the input values as scores to maximize;
            they will be converted to costs internally.

    Returns:
        A list of (row_index, col_index) pairs representing the assignment.

    Notes:
        - Tries to use scipy.optimize.linear_sum_assignment first. If SciPy is
          not available, falls back to the `munkres` package. If neither is
          available an ImportError is raised with instructions to install one
          of them (e.g. `pip install scipy`).
        - Supports rectangular matrices. If `munkres` is used it pads the
          matrix to square shape with large costs.
    """
    # Extract raw 2D list
    if hasattr(matrix, "data"):
        data = matrix.data
    else:
        data = matrix

    # Validate
    if not data:
        return []
    rows = len(data)
    cols = len(data[0]) if rows > 0 else 0

    # If either side is empty, no assignments are possible
    if rows == 0 or cols == 0:
        return []

    # Try SciPy first
    # Try SciPy first (if available). Use numpy arrays and replace infinities
    # with a large finite cost so the solver doesn't choke on inf values.
    try:
        from scipy.optimize import linear_sum_assignment
        import numpy as _np

        if maximize:
            max_val = max(max(row) for row in data)
            cost = _np.array([[max_val - float(v) for v in row] for row in data], dtype=float)
        else:
            cost = _np.array([[float(v) for v in row] for row in data], dtype=float)

        # Replace infinities with a large finite cost (larger than any finite cost in matrix)
        finite_mask = _np.isfinite(cost)
        if _np.any(finite_mask):
            large_cost = _np.nanmax(_np.abs(cost[finite_mask])) * 1e6 + 1.0
        else:
            large_cost = 1e12
        cost[~finite_mask] = large_cost

        row_ind, col_ind = linear_sum_assignment(cost)
        return list(zip(list(row_ind), list(col_ind)))
    except ImportError:
        # SciPy not installed -> fall through to munkres fallback
        pass
    except Exception:
        # SciPy present but solver failed on this input; try fallback
        pass

    # Fall back to munkres (if available)
    try:
        from munkres import Munkres

        m = Munkres()

        # munkres expects a square matrix and finite numeric costs. Build the
        # base cost matrix and pad to square with a large cost value.
        if maximize:
            max_val = max(max(row) for row in data)
            base = [[max_val - float(v) for v in row] for row in data]
        else:
            base = [[float(v) for v in row] for row in data]

        n = max(rows, cols)
        # Choose a large padding cost
        # If any finite values exist, base_max is their max abs; else use 1e6
        # Build list of finite absolute values safely
        flat_vals = []
        for row in base:
            for v in row:
                try:
                    fv = float(v)
                except Exception:
                    continue
                if fv == float('inf') or _isnan(fv):
                    continue
                flat_vals.append(abs(fv))

        pad_cost = (max(flat_vals) * 1e6 + 1.0) if flat_vals else 1e12

        # Build padded square matrix
        padded = [row + [pad_cost] * (n - cols) for row in base]
        for _ in range(n - rows):
            padded.append([pad_cost] * n)

        indexes = m.compute(padded)
        result = [(r, c) for r, c in indexes if r < rows and c < cols]
        return result
    except ImportError:
        raise ImportError(
            "Hungarian matching requires `scipy` or `munkres`. "
            "Install one with: pip install scipy  # or pip install munkres"
        )
    except Exception:
        # If both approaches fail for unexpected reasons, surface an informative error
        raise RuntimeError("Hungarian matching failed with both scipy and munkres fallbacks.")


def _isnan(x):
    try:
        return x != x
    except Exception:
        return False