"""
config/config.py
Loads config.yaml and exposes a single dict `CFG`.
"""
import yaml
import os


def load_config(path: str = None) -> dict:
    """Load configuration from YAML file. Defaults to <project_root>/config.yaml."""
    if path is None:
        here = os.path.dirname(__file__)
        path = os.path.join(here, "..", "config.yaml")
    path = os.path.abspath(path)
    with open(path, "r") as f:
        return yaml.safe_load(f)


CFG = load_config()
