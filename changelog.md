# Changelog

The author somehow decides to maintain a changelog starting from version 0.3.3 !!

## [0.3.8] - 2026-05-03

### Added

- Added CatBoostRegressor support for the Python package.
- Added optional install extras for XGBoost, LightGBM, CatBoost, and all boosting backends.

### Modified

- Improved the C++ weight traversal by replacing per-branch vector copies with in-place backtracking.
- Made model-library imports optional and clearer.

## [0.3.3] - 2024-10-07

### Added

- Introduce the support for LightGBM !!!!
- Added a formatter that transforms LightGBM dataframe to a nice format that can feed to qshap !!
- Added another formatter that transform the above formatter that enables calculation of Shapley value for each tree from LightGBM !!

## [0.3.4] - 2024-12-09

### Added

- Cumulative R-squared plot !!!

## [0.3.5] - 2025-05-01

### Modified

- init_prediction in function `summarize_tree` and `T2` so that the functions are more extensible
- Add values and n_node_samples to `tree_summary` class
