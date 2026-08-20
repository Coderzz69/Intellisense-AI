"""
config/config.py
Loads config.yaml and exposes a single dict `CFG`.
"""
import yaml
import os


def load_config(path: str = None) -> dict:
    if path is None:
        # Walk up from this file's directory to find config.yaml at project root.
        here = os.path.dirname(__file__)
        path = os.path.join(here, "..", "config.yaml")
    path = os.path.abspath(path)
    with open(path, "r") as f:
        return yaml.safe_load(f)


CFG = load_config()
