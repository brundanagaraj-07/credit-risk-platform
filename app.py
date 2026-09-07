import runpy
import sys
from pathlib import Path

# Ensure project root is in python search path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Run ui/app.py
ui_app_path = ROOT_DIR / "ui" / "app.py"
runpy.run_path(str(ui_app_path), run_name="__main__")
