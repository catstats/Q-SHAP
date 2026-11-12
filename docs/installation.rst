Installation and supported environments
=======================================

The development source declares Python >=3.9,<3.15 and NumPy <2.5.
Python 3.13 and 3.14 validation details are recorded in
``docs/PYTHON_COMPATIBILITY.md``. Python 3.15 is outside the validated range.
The published 2.0.0 release retains the older Python <3.13 metadata until
the development changes are released.

From a development checkout, use the intended interpreter consistently:

.. code-block:: sh

   python -m pip install --upgrade pip
   python -m pip install ".[all]"

For a published release, use ``python -m pip install qshap`` (or the model
extras). A compatible wheel avoids local compilation; a source installation
requires a C++17 compiler and the headers for that exact Python interpreter.

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
