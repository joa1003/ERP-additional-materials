#!/usr/bin/env python3
from pathlib import Path
import subprocess
import sys

module_path = Path(__file__).resolve().parents[1] / "src" / "survey_group_composition_survey_group_composition.py"
subprocess.run([sys.executable, str(module_path), "--selftest"], check=True)
