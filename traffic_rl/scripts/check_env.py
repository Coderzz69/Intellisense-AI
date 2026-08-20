"""
scripts/check_env.py
Phase 0 smoke test — verifies all dependencies are importable.
"""
import sys
import os
import importlib

# Add SUMO tools to path before importing traci
_sumo_home = os.environ.get("SUMO_HOME", r"C:\Program Files (x86)\Eclipse\Sumo")
_sumo_tools = os.path.join(_sumo_home, "tools")
if _sumo_tools not in sys.path:
    sys.path.insert(0, _sumo_tools)


def check(name, fn):
    try:
        fn()
        print(f"  [OK]  {name}")
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        return False
    return True


ok = True
ok &= check("torch",        lambda: importlib.import_module("torch"))
ok &= check("numpy",        lambda: importlib.import_module("numpy"))
ok &= check("yaml",         lambda: importlib.import_module("yaml"))
ok &= check("matplotlib",   lambda: importlib.import_module("matplotlib"))
ok &= check("tqdm",         lambda: importlib.import_module("tqdm"))
ok &= check("traci",        lambda: importlib.import_module("traci"))

print()
if ok:
    import torch
    print(f"All OK  (torch={torch.__version__}, "
          f"CUDA={'yes' if torch.cuda.is_available() else 'no'})")
else:
    print("Some checks FAILED — see above.")
    sys.exit(1)
