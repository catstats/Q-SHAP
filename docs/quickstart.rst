Quick start
===========

This example uses scikit-learn and the default single process. Installing
``qshap[xgboost]``, ``qshap[lightgbm]``, or ``qshap[catboost]`` enables the
corresponding optional regression model libraries.

.. code-block:: python

   import numpy as np
   from sklearn.ensemble import GradientBoostingRegressor
   from qshap import gazer, vis

   rng = np.random.default_rng(42)
   x = rng.normal(size=(100, 3))
   y = x[:, 0] - x[:, 1] + rng.normal(scale=0.1, size=100)
   model = GradientBoostingRegressor(n_estimators=10, max_depth=2).fit(x, y)
   explainer = gazer(model)
   result = explainer.rsq(x, y, local=True)
   np.testing.assert_allclose(result.local_rsq.sum(axis=0), result.rsq)
   summary = result.rsq
   first_tree = explainer.get_tree(0)
   figure = vis.heatmap(result, n_show=20, show=False)
   figure.savefig("local_contributions.pdf")

The result contains the global vector ``rsq`` and matrices ``loss`` and
``local_rsq``. The latter equals ``-loss / sum((y - mean(y)) ** 2)``;
its columns sum to global contributions. Negative contributions are valid.

Parallel scripts
----------------

Use a main guard on platforms that start workers by spawning Python:

.. code-block:: python

   def main():
       # Load data, fit model, and create explainer inside this function.
       result = explainer.rsq(x, y, local=True, ncore=2)

   if __name__ == "__main__":
       main()

Replace the commented preparation with your own data/model setup. The
parallel bar shows completed chunks. It appears before worker startup;
the first update follows a completed chunk, including its import and
compilation costs. Compare with ``ncore=1`` before increasing workers.
Notebook execution depends on the platform's multiprocessing support.
