Supported model examples
========================

The public workflow is the same for every supported regression model:
fit a model, create ``gazer(model)``, then call ``rsq(x, y)``.
The following examples use scalar responses and numeric input columns.

Run the shared data cell, choose one model block, and finish with the
decomposition cell. Each downloadable script includes all three steps.

Shared data
-----------

.. literalinclude:: scripts/decision_tree.py
   :language: python
   :start-after: # example:data
   :end-before: # end:data

Scikit-learn decision tree
--------------------------

.. literalinclude:: scripts/decision_tree.py
   :language: python
   :start-after: # example:model
   :end-before: # end:model

:download:`Download decision_tree.py <scripts/decision_tree.py>`.

Scikit-learn gradient boosting
------------------------------

.. literalinclude:: scripts/gradient_boosting.py
   :language: python
   :start-after: # example:model
   :end-before: # end:model

:download:`Download gradient_boosting.py <scripts/gradient_boosting.py>`.

XGBoost
-------

Install its optional backend with ``python -m pip install "qshap[xgboost]"``.
Pass the fitted ``XGBRegressor`` wrapper to Q-SHAP.

.. literalinclude:: scripts/xgboost_example.py
   :language: python
   :start-after: # example:model
   :end-before: # end:model

:download:`Download xgboost_example.py <scripts/xgboost_example.py>`.

LightGBM
--------

Install its optional backend with ``python -m pip install "qshap[lightgbm]"``.

.. literalinclude:: scripts/lightgbm_example.py
   :language: python
   :start-after: # example:model
   :end-before: # end:model

:download:`Download lightgbm_example.py <scripts/lightgbm_example.py>`.

CatBoost
--------

Install its optional backend with ``python -m pip install "qshap[catboost]"``.
See :doc:`../installation` for Python 3.15 source-build requirements.

.. literalinclude:: scripts/catboost_example.py
   :language: python
   :start-after: # example:model
   :end-before: # end:model

:download:`Download catboost_example.py <scripts/catboost_example.py>`.

The default symmetric-tree model uses the specialized global algorithm.
Numeric ``Depthwise`` and ``Lossguide`` models use the general-tree path.
Categorical/CTR, text, and embedding split representations are unsupported.

Compute contributions with any of these models
----------------------------------------------

.. literalinclude:: scripts/decision_tree.py
   :language: python
   :start-after: # example:compute
   :end-before: # end:compute

Keep the feature matrix in its training column order. Contributions from
different fitted models describe those respective models, so their values
need not agree. See :doc:`local_contributions` to request local output.
