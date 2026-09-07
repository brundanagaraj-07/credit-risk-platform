import os
import runpy
import sys
from pathlib import Path

# Ensure project root is at index 0 of sys.path
ROOT_DIR = Path(__file__).resolve().parent
root_str = str(ROOT_DIR)
if root_str not in sys.path:
    sys.path.insert(0, root_str)

# Ensure working directory matches ROOT_DIR
os.chdir(root_str)

# Run ui/app.py with the root path preserved
ui_app_path = ROOT_DIR / "ui" / "app.py"
runpy.run_path(str(ui_app_path), run_name="__main__")
