Installation and supported environments
=======================================

qshap 3.0.0 supports Python >=3.9,<3.16 and NumPy <2.6. Python 3.15 requires
Numba >=0.68; validation used NumPy 2.5.4. Exact interpreter versions and
backend coverage are recorded in ``docs/PYTHON_COMPATIBILITY.md``. Standard
CPython builds with the GIL are covered; free-threaded builds have not
been validated.

Install from PyPI (recommended)
-------------------------------

.. code-block:: sh

   python -m pip install qshap
   python -m pip install "qshap[xgboost,lightgbm]"

Install from local sources
---------------------------

From the Python package source directory:

.. code-block:: sh

   python -m pip install .
   python -m pip install ".[xgboost,lightgbm]"

Alternatively, from the directory containing the source archive:

.. code-block:: sh

   python -m pip install "./qshap-3.0.0.tar.gz[xgboost,lightgbm]"

A compatible wheel avoids local compilation. A source installation requires
a C++17 compiler and the headers for that exact Python interpreter.

CatBoost and Python 3.15
------------------------

CatBoost 1.2.10 has no CPython 3.15 wheel on PyPI at the time of validation.
For a precompiled CatBoost installation, use Python 3.14 and install
``.[catboost]`` or ``.[all]``. On Python 3.15 these extras attempt an upstream
source build. That requires additional build tools (including Ninja,
CMake, Conan, and a suitable compiler), downloads native dependencies, and
can take substantially longer than installing the other backends.
The numerical CatBoost backend was validated on Python 3.15 using an
unmodified, source-built CPU package with the optional Jupyter visualization
widget disabled (``--no-widget``). The compatibility record gives the full
build command. See also the
`upstream Python 3.15 issue <https://github.com/catboost/catboost/issues/3213>`_.

Missing Python.h on Ubuntu or Debian
------------------------------------

``Python.h`` is provided by the interpreter's development installation.
A virtual environment does not install missing system headers. PEP 517
build isolation installs Python build dependencies, not operating-system
compiler/header packages.

For the distribution's default Python, install its matching tools before
creating the environment:

.. code-block:: sh

   sudo apt update
   sudo apt install build-essential python3-dev python3-venv
   python3 -m venv .venv
   . .venv/bin/activate
   python -m pip install --upgrade pip
   python -m pip install .

Ubuntu 26.04's ``python3-dev`` targets Python 3.14, so it does not supply
headers for a separately installed Python 3.12. For Python 3.12, install
``python3.12-dev`` only if the repository/provider supplying that interpreter
also provides it. Otherwise obtain a complete development installation from
that provider, or use the distribution's supported default interpreter.
Do not point Python 3.12 at Python 3.14's headers.

Inspect the active interpreter and the expected header location:

.. code-block:: sh

   python -c "import sys,sysconfig; from pathlib import Path; p=Path(sysconfig.get_path('include'))/'Python.h'; print(sys.executable); print(p); print(p.is_file())"
   python -m pip wheel --no-deps --wheel-dir dist .

Q-SHAP now checks for headers before invoking its C++ compiler and reports
the active interpreter and searched paths if headers are absent. The
matching compiler/headers are still required. This guidance applies both
to installing a local source checkout and building a source release from pip.

See the `Python documentation <https://docs.python.org/3.12/using/unix.html#python-related-paths-and-files>`_
and Ubuntu's `python3-dev package <https://packages.ubuntu.com/resolute/python3-dev>`_.

Build this documentation
------------------------

.. code-block:: sh

   python -m pip install ".[docs]"
   python -m sphinx -W --keep-going -b html docs docs/_build/html

Open ``docs/_build/html/index.html`` locally. The reference is generated
from the public classes' docstrings using Sphinx autodoc and Napoleon;
no hosted documentation is assumed.
