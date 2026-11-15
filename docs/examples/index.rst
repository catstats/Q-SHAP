Examples
========

These walkthroughs show how to use Q-SHAP for specific tasks. Start with
the global decomposition, then request observation-level output or choose
another supported model. Each page links to complete Python scripts.

* :doc:`global_rsq`

  Fit a model, compute feature contributions, and inspect a stored tree.

* :doc:`local_contributions`

  Relate local matrices to global contributions and select heatmap rows.

* :doc:`model_backends`

  Use scikit-learn, XGBoost, LightGBM, or CatBoost through one interface.

* :doc:`plotting`

  Label, rank, plot, and save contributions.

* :doc:`parallel`

  Run a complete script with multiple workers and preserve row order.

The synthetic examples create their data locally. Optional model libraries
and their installation requirements are described in :doc:`../installation`.
For parameter definitions and return types, see :doc:`../api`.

.. toctree::
   :maxdepth: 1
   :hidden:

   global_rsq
   local_contributions
   model_backends
   plotting
   parallel
