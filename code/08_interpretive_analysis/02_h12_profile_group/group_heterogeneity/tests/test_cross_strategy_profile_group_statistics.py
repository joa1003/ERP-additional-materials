import importlib.util
from pathlib import Path

MODULE = Path(__file__).resolve().parents[1] / "src/cross_strategy_profile_group_cross_strategy_profile_group_appendix_closure.py"
spec = importlib.util.spec_from_file_location("cross_strategy_profile_group", MODULE)
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
mod.run_self_test()
