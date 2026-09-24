"""uvcorr - RENA-3 fine-timing U/V ellipse correction from raw PET acquisition files.

Fits a least-squares ellipse to each active channel's (U, V) quadrature points,
read directly from a raw ``.dat`` acquisition through a sibling HDF5 UV cache,
maps it to a circle, and exports the RadialAnalysis-format ``.tec`` file plus a
``radial_summary.csv``. A PyQt6/pyqtgraph GUI (``uvcorr-gui``) views and
re-fits the results. Built on the adc2kev library (parser and geometry).

See ``docs/planning/UV_ELLIPSE_CORRECTION_PLAN.md`` for the design.
"""

__version__ = "0.1.0"
