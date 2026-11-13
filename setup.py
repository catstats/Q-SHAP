import numpy as np
import sys
import sysconfig
from pathlib import Path
from setuptools import Extension, setup, find_packages
from setuptools.command.build_ext import build_ext
from setuptools.errors import CompileError


def check_python_headers(include_dirs):
    """Fail with interpreter-specific guidance before invoking the compiler."""
    paths = list(include_dirs or [])
    paths.extend(sysconfig.get_paths().get(key) for key in ("include", "platinclude"))
    paths = list(dict.fromkeys(str(path) for path in paths if path))
    if not any((Path(path) / "Python.h").is_file() for path in paths):
        version = f"{sys.version_info.major}.{sys.version_info.minor}"
        raise CompileError(
            f"Python.h was not found for Python {version} ({sys.executable}). "
            f"Searched: {', '.join(paths)}. Install development headers matching "
            "this interpreter and a C++17 compiler, then retry with this "
            "interpreter's `python -m pip install .`. On Debian/Ubuntu, use "
            f"build-essential and python{version}-dev if that interpreter is "
            "provided by your configured repositories. For a separately "
            "installed Python, obtain its headers from the same provider; "
            "python3-dev may target a different Python version. "
            "See docs/installation.rst."
        )


class BuildExt(build_ext):
    def build_extensions(self):
        check_python_headers(
            list(self.include_dirs or []) + list(self.compiler.include_dirs or [])
        )
        opts = {
            "msvc": ["/O2", "/std:c++17"],
            "unix": ["-O3", "-std=c++17"],
        }
        for ext in self.extensions:
            ext.extra_compile_args = opts.get(self.compiler.compiler_type, [])
        super().build_extensions()


qshap_cpp = Extension(
    "qshap._qshap_cpp",
    sources=["qshap/_qshap_cpp.cpp"],
    depends=["qshap/catboost_fused_router.h"],
    include_dirs=[np.get_include()],
    language="c++",
)

setup(
    name='qshap',
    version='2.0.0',
    description='Exact computation of Shapley R-squared for tree ensembles in polynomial time',
    long_description=(Path(__file__).parent / 'README.md').read_text(encoding='utf-8'),
    long_description_content_type='text/markdown',  # Specify that the long_description is in Markdown
    author='Zhongli Jiang, Dabao Zhang', 
    author_email='jiang548@purdue.edu, zdb969@hs.uci.edu',
    license="GPL-2.0",
    packages=find_packages(),
    package_data={"qshap": ["*.h"]},
    ext_modules=[qshap_cpp],
    cmdclass={"build_ext": BuildExt},
    url="https://github.com/catstats/Q-SHAP",
    install_requires=[
        'numpy>=1.22,<2.6',
        'scikit-learn',
        'shap',
        'numba; python_version < "3.15"',
        'numba>=0.68; python_version >= "3.15"',
        'ipywidgets',
        'pandas',
        'matplotlib',
    ],
    extras_require={
        'xgboost': ['xgboost'],
        'lightgbm': ['lightgbm'],
        'catboost': ['catboost'],
        'all': ['xgboost', 'lightgbm', 'catboost'],
        'docs': ['sphinx>=7,<10'],
    },
    classifiers=[
        'Operating System :: Microsoft :: Windows',
        'Operating System :: Unix',
        'Operating System :: MacOS',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        'Programming Language :: Python :: 3.14',
        'Programming Language :: Python :: 3.15',
    ],
    python_requires='>=3.9,<3.16',
    zip_safe=False
)
