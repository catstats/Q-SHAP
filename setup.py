import numpy as np
from setuptools import Extension, setup, find_packages
from setuptools.command.build_ext import build_ext


class BuildExt(build_ext):
    def build_extensions(self):
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
    include_dirs=[np.get_include()],
    language="c++",
)

setup(
    name='qshap',
    version='0.3.7',
    description='Exact computation of Shapley R-squared for tree ensembles in polynomial time',
    long_description=open('README.md').read(),    
    long_description_content_type='text/markdown',  # Specify that the long_description is in Markdown
    author='Zhongli Jiang, Dabao Zhang', 
    author_email='jiang548@purdue.edu, zdb969@hs.uci.edu',
    license="GPL-2.0",
    packages=find_packages(),
    ext_modules=[qshap_cpp],
    cmdclass={"build_ext": BuildExt},
    url="https://github.com/catstats/Q-SHAP",
    install_requires=['numpy', 'scikit-learn',  'shap', 'numba', 'ipywidgets', 'pandas', 'matplotlib'],
    extras_require={'xgboost': ['xgboost'],'lightgbm': ['lightgbm']},
    classifiers=[
        'Operating System :: Microsoft :: Windows',
        'Operating System :: Unix',
        'Operating System :: MacOS',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
    python_requires='>=3.7', 
    zip_safe=False
)
