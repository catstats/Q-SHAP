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

The October run did not test Python 3.15 or NumPy 2.5. At that time the
source declared `>=3.9,<3.15` and `numpy<2.5`. The later validation below
supersedes those bounds.


## Additional validation on 2026-11-13

The current implementation at `d028ec4`, with the proposed Python/NumPy
metadata and NumPy guard changes, was built and installed separately in
each interpreter environment. Tests ran outside the source checkout on
macOS 15.5 / Apple Silicon, with standard GIL-enabled CPython. The compiled
extension was required to load, so a fallback could not mask a missing C++
build. No numerical algorithm changes were needed.

| Python | Passed | Failed/errors | Reported skips | C++ backend |
| --- | ---: | ---: | ---: | --- |
| 3.12.16 | 56 | 0 | 0 | Built, loaded, and exercised |
| 3.13.18 | 56 | 0 | 0 | Built, loaded, and exercised |
| 3.14.10 | 56 | 0 | 0 | Built, loaded, and exercised |
| 3.15.2 | 56 | 0 | 0 | Built, loaded, and exercised |

Dependency consistency checks passed in each environment. The tested stack
was NumPy 2.5.4, Numba 0.68.1, llvmlite 0.50.0, SHAP 0.54.1,
scikit-learn 1.9.1, XGBoost 3.4.1, LightGBM 4.7.0, pandas 3.1.1,
Matplotlib 3.11.2, and pytest 9.2.0. CatBoost 1.2.10 was installed for
the full-backend environments. Native OpenMP was supplied to these temporary
processes through a matching ARM64 `libomp.dylib` and `DYLD_LIBRARY_PATH`.

The original `numpy<2.5` constraint prevented a wheel-based Python 3.15
installation: its available NumPy wheels started in the 2.5 series. The
tested source now permits `numpy<2.6`, declares Python `>=3.9,<3.16`, and
requires Numba >=0.68 specifically on Python 3.15. The runtime NumPy guard
matches the new dependency range.

### CatBoost availability

CatBoost 1.2.10 provides no cp315 wheel on PyPI, and its
[upstream Python 3.15 support issue](https://github.com/catboost/catboost/issues/3213)
is open. The first standard source-build attempt also lacked Ninja; the
test then installed Ninja in the temporary environment and retried with
a separate temporary Conan directory. The final outcome is recorded below.

The unmodified CatBoost 1.2.10 native CPU extension compiled successfully
against Python 3.15.2. Default wheel packaging then failed at its optional
Jupyter widget step because Yarn was absent. Rebuilding with the officially
supported `--no-widget` option and four parallel build jobs produced an
installable cp315 ARM64 wheel. After installing that wheel, all 56 Q-SHAP
tests passed with zero failures and zero skips, including CatBoost TreeSHAP,
general/symmetric trees, missing routing, and C++/Numba comparisons. CUDA
and CatBoost's optional Jupyter visualization widget were not part of this
CPU validation. No third-party source code was patched.

Activate the Python 3.15 virtual environment before running the effective
build and installation commands:

```sh
python -m pip install ninja
python -m pip wheel --no-deps --no-binary=catboost \
  --config-settings=--build-option=--no-widget \
  --config-settings=--build-option=--parallel=4 \
  --wheel-dir /tmp/catboost-cp315 catboost==1.2.10
python -m pip install /tmp/catboost-cp315/catboost-1.2.10-cp315-*.whl
```

An appropriate C/C++ toolchain is also required; our test used Apple Clang
17 and Xcode's SDK. Conan used its own temporary directory, configured
through `CONAN_HOME`. The installed CPU wheel's SHA-256 was
`9a241213cb13eb3a3098c358cac807eb7f61b91ed0547e6c1dd955441b671f22`.
This is a local source-build result, not a claim that upstream has released
Python 3.15 wheels.

### Reproduce the Python 3.15 core/model tests

```sh
QSHAP_SOURCE="$(pwd)"
python3.15 -m venv /tmp/qshap-py315
/tmp/qshap-py315/bin/python -m pip install --upgrade pip
/tmp/qshap-py315/bin/python -m pip install ".[xgboost,lightgbm]" pytest setuptools
/tmp/qshap-py315/bin/python -m pip check
cd /tmp
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 MPLBACKEND=Agg \
  /tmp/qshap-py315/bin/python -m pytest "$QSHAP_SOURCE/tests" --import-mode=importlib -ra
```

To reproduce all 56 cases on Python 3.15, install the source-built CatBoost
wheel described above into the same `/tmp/qshap-py315` environment before
invoking pytest. Without it, the core/model run reports CatBoost skips.

Supply an appropriate native OpenMP runtime on macOS. For the Python
3.12–3.14 full runs, use the corresponding interpreter and install `.[all]`.
The CI matrix includes 3.12–3.15; its 3.15 job installs XGBoost and LightGBM
and explicitly leaves native CatBoost coverage to the 3.12–3.14 jobs while
cp315 wheels are unavailable. Linux CI has been configured, not run locally.
Windows, Intel macOS, and free-threaded CPython were not validated here.


## Ubuntu 26.04 source installation on 2026-11-16

The local qshap 3.0.0 source at `d64f2ab` was installed in fresh Ubuntu
26.04.1 containers with CPython 3.12.16 from the deadsnakes PPA and GCC
15.2.0. ARM64 ran natively on Apple Silicon; x86-64 ran under Docker's
QEMU emulation. The scope was source installation and a small compiled
decision-tree calculation.

| Architecture | Without Python 3.12 headers | After installing `python3.12-dev` |
| --- | --- | --- |
| x86-64 | Expected missing-header error and interpreter-specific guidance | Source build, extension import, and C++ calculation succeeded |
| ARM64 | Expected missing-header error and interpreter-specific guidance | Source build, extension import, and C++ calculation succeeded |

Each environment first verified that `/usr/include/python3.12/Python.h`
was absent. Installing the matching development package from the same
provider made the header available; the same unmodified package source
then built successfully. The runtime check required the installed Linux
ELF extension to load and exercised `backend="cpp"` outside the source
directory. The 32-observation, four-feature decision-tree example verified
finite contributions, local-to-global column sums, and agreement between
the contribution total and the fitted model's training R-squared.

Both environments used NumPy 2.5.5, Numba 0.68.1, SHAP
0.54.1, and scikit-learn 1.9.1. The full model regression
suite and paper benchmarks were not run in these containers. The editor's
interpreter provider and exact dependency versions were unspecified; this
test reproduces the missing-header condition and verifies its remedy.
