Global feature contributions
============================

How much does each feature contribute to the fitted model's R-squared?
This example fits a small regression ensemble and returns one contribution
per input column. All computations use the same 80 observations used to fit
the model, so this is an in-sample example.

:download:`Download the complete script <scripts/global_rsq.py>`.

Fit a model
-----------

The response depends on three of four numeric features. A fixed random seed
makes the generated dataset reproducible.

.. literalinclude:: scripts/global_rsq.py
   :language: python
   :start-after: # example:setup
   :end-before: # end:setup

Compute the decomposition
-------------------------

Create an explainer once and pass the feature matrix and response to
:py:meth:`qshap.gazer.rsq`. The returned vector follows the original column
order. The script prints its sum alongside the fitted model's R-squared.

.. literalinclude:: scripts/global_rsq.py
   :language: python
   :start-after: # example:decompose
   :end-before: # end:decompose

Keep this vector signed: a feature contribution can be negative. It describes
the feature's allocation in the model-performance decomposition; it does
not establish a causal effect.

Attach feature names and rank values
------------------------------------

Sort the vector and its feature labels with the same index array.

.. literalinclude:: scripts/global_rsq.py
   :language: python
   :start-after: # example:rank
   :end-before: # end:rank

Inspect one tree
----------------

:py:meth:`qshap.gazer.get_tree` exposes the stored fields of a parsed tree.
Python tree indices start at zero.

.. literalinclude:: scripts/global_rsq.py
   :language: python
   :start-after: # example:tree
   :end-before: # end:tree

Continue with :doc:`local_contributions` to see how observations contribute
to each global value, or :doc:`plotting` to visualize the result.
