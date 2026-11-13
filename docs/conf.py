import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
project = "Q-SHAP"
author = "Zhongli Jiang and Dabao Zhang"
release = "3.0.0"
extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon", "sphinx.ext.viewcode"]
html_theme = "alabaster"
autodoc_member_order = "bysource"
exclude_patterns = ["_build"]
