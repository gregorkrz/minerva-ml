"""Evaluation scripts: data collection and plotting for classification and regression.

Workflow (Python 3.10+ recommended)::

    python -m src.eval.collect_eval_data --flag Run_2703
    python -m src.eval.plot_steps
    python -m src.eval.plot_regression
    python -m src.eval.plot_classification_W
    python -m src.eval.plot_classification_q3
    python -m src.eval.plot_classification_Pions

Run from the ``minerva-data-processing`` repository root with ``PYTHONPATH=``.

Pickles are read from ``--out-dir`` (default ``out/`` under the repo). Plot PDFs
are written to ``--plots-dir`` (default ``plots/`` under the repo), so scratch
paths like ``--out-dir /pscratch/.../eval_runs`` can pair with local ``plots/``.

To stage PDFs for LaTeX with stable filenames, run
``python -m src.scripts.copy_figures_for_paper`` (after the plot steps above).
"""
