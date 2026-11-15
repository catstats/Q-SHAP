Public Python API
=================

The package exports ``gazer`` and ``vis``. Arrays use the training feature
order; tree and observation indices are zero-based. Internal modules and the
compiled extension are implementation details.

For complete workflows and downloadable scripts, see :doc:`examples/index`.

Model explainer
---------------

Examples: :doc:`examples/global_rsq`, :doc:`examples/local_contributions`,
:doc:`examples/model_backends`, and :doc:`examples/parallel`.

.. autoclass:: qshap.gazer
   :members: get_tree, loss, rsq, gcorr

Visualization
-------------

Examples: :doc:`examples/plotting` and :doc:`examples/local_contributions`.

.. autoclass:: qshap.vis
   :members: rsq, loss, heatmap, loss_heatmap, elbow, cumu, gcorr
