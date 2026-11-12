Public Python API
=================

The package exports ``gazer`` and ``vis``. Arrays use the training feature
order; tree and observation indices are zero-based. Internal modules and the
compiled extension are implementation details.

Model explainer
---------------

.. autoclass:: qshap.gazer
   :members: get_tree, loss, rsq, gcorr

Visualization
-------------

.. autoclass:: qshap.vis
   :members: rsq, loss, heatmap, loss_heatmap, elbow, cumu, gcorr
