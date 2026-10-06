"""Explicit paths supplied by the figure runner; no original workstation paths."""
import os
from pathlib import Path

if not os.environ.get('TARF_FIGURE_OUTPUT'):
    raise RuntimeError('Use scripts/render_figures.py, not the internal plotting modules')
OUTPUT = Path(os.environ['TARF_FIGURE_OUTPUT']).resolve()
SOURCE_DATA = OUTPUT / 'source_data'
ASSETS = Path(os.environ['TARF_GDC_ASSETS']).resolve()
