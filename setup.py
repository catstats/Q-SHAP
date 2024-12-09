from setuptools import setup, find_packages

setup(
    name='qshap',
    version='0.3.5',
    description='Exact computation of Shapley R-squared for tree ensembles in polynomial time',
    long_description=open('README.md').read(),    
    long_description_content_type='text/markdown',  # Specify that the long_description is in Markdown
    author='Zhongli Jiang, Dabao Zhang', 
    author_email='jiang548@purdue.edu, zdb969@hs.uci.edu',
    license="GPL-2.0",
    packages=find_packages(),
    url="https://github.com/catstats/Q-SHAP",
    setup_requires=['numpy'],
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
