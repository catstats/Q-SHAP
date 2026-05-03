from qshap.main import gazer

__version__ = "0.3.8"


def __getattr__(name):
    if name == "vis":
        from qshap.vis_module import vis
        return vis
    raise AttributeError(f"module 'qshap' has no attribute {name!r}")


__all__ = ["gazer", "vis"]
