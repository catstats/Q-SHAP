Local contributions and heatmaps
================================

Which observations contribute to a feature's global R-squared allocation?
Request the local matrices, then use a signed heatmap to inspect individual
rows. The dataset and fitted model match :doc:`global_rsq`.

:download:`Download the complete script <scripts/local_contributions.py>`.

Prepare the data and model
--------------------------

.. literalinclude:: scripts/local_contributions.py
   :language: python
   :start-after: # example:setup
   :end-before: # end:setup

Understand the three returned quantities
----------------------------------------

With ``local=True``, :py:meth:`qshap.gazer.rsq` returns a namespace with:

* ``rsq``: the global contribution vector, with one value per feature.
* ``loss``: an observation-by-feature matrix of raw loss contributions.
* ``local_rsq``: the same matrix after sign reversal and normalization by
  the full evaluated sample's total sum of squares.

.. math::

   \operatorname{local\_rsq}_{ij}
   = -\frac{\operatorname{loss}_{ij}}{\sum_k (y_k-\bar y)^2}.

.. math::

   \operatorname{rsq}_j=\sum_i\operatorname{local\_rsq}_{ij}.

.. literalinclude:: scripts/local_contributions.py
   :language: python
   :start-after: # example:local
   :end-before: # end:local

Rows contribute to the global decomposition. A row total is not a separate
R-squared statistic for that observation. Without subsampling, matrix rows
retain the order of the supplied data.

Plot observations with large positive and negative totals
---------------------------------------------------------

:py:meth:`qshap.vis.heatmap` orders feature columns by global contribution.
Automatic row selection takes observations from both extremes of the signed
row totals. The separate ``Total`` column is the sum across each displayed row.

.. literalinclude:: scripts/local_contributions.py
   :language: python
   :start-after: # example:heatmap
   :end-before: # end:heatmap

``show=False`` returns a Matplotlib figure for saving or caller-managed
display. Here the figure is saved as a PDF and then closed.

Select particular observations
------------------------------

Provide observation IDs and select rows explicitly to keep a chosen order.
Feature names and observation IDs must match the full matrix dimensions.

.. literalinclude:: scripts/local_contributions.py
   :language: python
   :start-after: # example:selection
   :end-before: # end:selection

Inspect raw loss contributions
------------------------------

Set ``quantity="loss"`` to display the raw matrix. This changes the sign and
scale relative to the normalized R-squared view.

.. literalinclude:: scripts/local_contributions.py
   :language: python
   :start-after: # example:loss
   :end-before: # end:loss

See :doc:`plotting` for global plots and :doc:`parallel` for local output
computed using multiple workers.
