import numpy as _np

__version__ = "2.0.0"


def _numpy_major_minor(version):
    parts = version.split(".")[:2]
    try:
        return tuple(int(part) for part in parts)
    except ValueError:
        return (0, 0)


if _numpy_major_minor(_np.__version__) >= (2, 5):
    raise ImportError(
        f"qshap {__version__} requires numpy<2.5 because the current numba/shap "
        "dependency stack does not support NumPy 2.5 yet."
    )


from qshap.main import gazer


def __getattr__(name):
    if name == "vis":
        from qshap.vis_module import vis
        return vis
    raise AttributeError(f"module 'qshap' has no attribute {name!r}")


__all__ = ["gazer", "vis"]
