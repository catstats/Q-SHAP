from setuptools import setup, find_packages

setup(
    name='qshap',
    version='0.1.0',
    description='Exact computation of shapley R-squared in polynomial time',
    long_description=open('README.md').read(),
    author='Zhongli Jiang',
    author_email='jiang548@purdue.edu',
    license="GPL-2.0",
    url='To be filled',
    packages=find_packages(),
    setup_requires=['numpy'],
    install_requires=['numpy', 'scikit-learn',  'shap', 'numba', 'ipywidgets', 'pandas', 'matplotlib'],
    classifiers=[
        'Operating System :: Microsoft :: Windows',
        'Operating System :: Unix',
        'Operating System :: MacOS',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    python_requires='>=3.7', 
    zip_safe=False
)
