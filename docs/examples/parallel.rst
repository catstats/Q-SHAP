Parallel scripts
================

This complete script uses two workers and compares their returned arrays
with serial execution. The tiny model illustrates the calling pattern;
worker startup can make this example slower than one worker.

Install the optional model backend:

.. code-block:: sh

   python -m pip install "qshap[xgboost]"

:download:`Download parallel.py <scripts/parallel.py>` and run it as a file:

.. code-block:: sh

   python parallel.py

.. literalinclude:: scripts/parallel.py
   :language: python

Why the main guard matters
--------------------------

Keep data creation, model fitting, and the call using multiple workers
inside ``main()``. The ``if __name__ == "__main__":`` guard prevents spawned
workers from starting the whole script again. Notebook multiprocessing
support varies by platform; use a script when trying parallel execution.

Progress and row order
----------------------

The progress bar appears before worker startup and advances as chunks
finish. The first completion includes startup, imports, and any compilation
cost. Chunks are reassembled in input order, so local matrices retain their
row alignment when subsampling is disabled.

Increase ``ncore`` only after comparing elapsed time on your own workload.
``-1`` requests available CPUs; worker counts are capped by CPU count and
the number of observations.

For symmetric CatBoost models, global-only output uses the specialized
cached algorithm independently of ``ncore``. See :doc:`model_backends` and
:py:meth:`qshap.gazer.rsq` for that path.
