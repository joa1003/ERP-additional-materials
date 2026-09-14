#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "survey_gain_associations_group_adjusted_survey_gain_associations.py"
spec = importlib.util.spec_from_file_location("survey_gain_associations_main", MODULE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError("Could not load Survey gain associations main module")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
module.selftest()
