# Python compatibility validation

Validated on 2026-10-18 using the public source at `25956be`, with the proposed
Python metadata update and the new `test_treeshap_compat.py` tests. No algorithm
changes were needed.

## Results

The tests ran against independently built and installed wheels, outside the
source directory, on macOS 15.5 / Apple Silicon (`arm64`), using standard
CPython builds with the GIL enabled.

| Python | Tests passed | Failed | Skipped | Compiled C++ backend |
| --- | ---: | ---: | ---: | --- |
| 3.12.12 | 52 | 0 | 0 | Loaded and exercised |
| 3.13.13 | 52 | 0 | 0 | Loaded and exercised |
| 3.14.4 | 52 | 0 | 0 | Loaded and exercised |

Dependency consistency checks also passed in all three environments. The same
dependency versions were installed for each interpreter:

| Dependency | Version |
| --- | --- |
| NumPy | 2.4.6 |
| Numba | 0.68.0 |
| llvmlite | 0.50.0 |
| SHAP | 0.54.1 |
| scikit-learn | 1.9.1 |
| XGBoost | 3.4.1 |
| LightGBM | 4.7.0 |
| CatBoost | 1.2.10 |
| pandas | 3.0.6 |
| Matplotlib | 3.11.2 |
| pytest | 9.1.1 |

The new tests check TreeSHAP additivity against model predictions, then compare
Q-SHAP's global R-squared, raw local loss, and local R-squared results between
the C++ and Numba backends. They cover scikit-learn decision trees and gradient
boosting, XGBoost, LightGBM, and CatBoost. The existing tests additionally cover
missing-value routing, split boundaries, CatBoost grow policies, parser
behavior, plotting, tree access, and API edge cases.

## Reproduce

Starting in a checkout containing this change, create an isolated environment
with the Python version to test. For example:

```sh
QSHAP_SOURCE="$(pwd)"
python3.14 -m venv /tmp/qshap-py314
/tmp/qshap-py314/bin/python -m pip install --upgrade pip
/tmp/qshap-py314/bin/python -m pip install ".[all]" pytest
/tmp/qshap-py314/bin/python -m pip check
cd /tmp
/tmp/qshap-py314/bin/python -c 'import xgboost, lightgbm, catboost; from qshap._backend import cpp_available; assert cpp_available()'
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  /tmp/qshap-py314/bin/python -m pytest "$QSHAP_SOURCE/tests" --import-mode=importlib -ra
```

On macOS, the model backends require a matching native OpenMP runtime. This run
used an existing ARM64 `libomp.dylib`, exposed only to the temporary test
processes through `DYLD_LIBRARY_PATH`. No system libraries were modified.

The initial Intel-interpreter installation attempt could not build llvmlite
without a separate LLVM toolchain. It is not counted as a successful platform
test. Windows, Linux, Intel macOS, and free-threaded CPython builds were not
tested locally; the added CI workflow will cover standard Python 3.12-3.14 on
Linux when run.

Python 3.15 and NumPy 2.5 were not tested in this run. The source therefore
expands the Python range to `>=3.9,<3.15` and retains `numpy<2.5`.
