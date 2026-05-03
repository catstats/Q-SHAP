import numpy as np

try:
    from qshap import _qshap_cpp
except Exception:  # pragma: no cover - exercised when extension is not built
    _qshap_cpp = None


def cpp_available():
    return _qshap_cpp is not None


def should_use_cpp(backend):
    if backend not in ("auto", "cpp", "numba"):
        raise ValueError("backend must be one of 'auto', 'cpp', or 'numba'")
    if backend == "numba":
        return False
    if _qshap_cpp is None:
        if backend == "cpp":
            raise RuntimeError(
                "The qshap C++ backend is not available. "
                "Build it with `python setup.py build_ext --inplace`, "
                "or use backend='numba'."
            )
        return False
    return True


def as_2d_float64(x):
    x_arr = np.asarray(x, dtype=np.float64, order="C")
    if x_arr.ndim == 1:
        x_arr = x_arr.reshape(1, -1)
    return x_arr


def as_target_vector(y, n_rows):
    y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
    if y_arr.shape[0] == 1 and n_rows != 1:
        y_arr = np.repeat(y_arr, n_rows)
    elif y_arr.shape[0] != n_rows:
        raise ValueError("y must have one value per row of x")
    return y_arr


def as_t0_matrix(t0_x):
    if isinstance(t0_x, list):
        t0_x = t0_x[0]
    t0_x = np.asarray(t0_x, dtype=np.float64, order="C")
    if t0_x.ndim == 1:
        t0_x = t0_x.reshape(1, -1)
    elif t0_x.ndim == 3 and t0_x.shape[-1] == 1:
        t0_x = np.ascontiguousarray(t0_x[:, :, 0], dtype=np.float64)
    return t0_x


def summary_tree_arrays(summary_tree):
    return (
        np.asarray(summary_tree.children_left, dtype=np.int64),
        np.asarray(summary_tree.children_right, dtype=np.int64),
        np.asarray(summary_tree.feature, dtype=np.int64),
        np.asarray(summary_tree.feature_uniq, dtype=np.int64),
        np.asarray(summary_tree.threshold, dtype=np.float64),
        np.asarray(summary_tree.sample_weight, dtype=np.float64),
        np.asarray(summary_tree.init_prediction, dtype=np.float64),
    )


def t2_cpp(x, summary_tree, store_v_invc, store_z):
    return _qshap_cpp.t2(
        x,
        *summary_tree_arrays(summary_tree),
        np.asarray(store_v_invc, dtype=np.complex128, order="C"),
        np.asarray(store_z, dtype=np.complex128, order="C"),
    )


def loss_treeshap_cpp(x, y, summary_tree, store_v_invc, store_z, t0_x, learning_rate):
    return _qshap_cpp.loss_treeshap(
        x,
        y,
        *summary_tree_arrays(summary_tree),
        np.asarray(store_v_invc, dtype=np.complex128, order="C"),
        np.asarray(store_z, dtype=np.complex128, order="C"),
        t0_x,
        learning_rate,
    )
