import runpy
from pathlib import Path

# Execute ui/app.py directly
ui_path = Path(__file__).resolve().parent / "ui" / "app.py"
runpy.run_path(str(ui_path), run_name="__main__")
