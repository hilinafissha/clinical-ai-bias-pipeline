"""
Configuration loader for the Clinical AI Bias Pipeline.
Ensures Kaggle and local environments use identical parameters and resolved paths.
"""

import os
import random
from types import SimpleNamespace
import numpy as np
import yaml

# Expected locations for config.yaml depending on the active environment
_DEFAULT_CANDIDATES = [
    "config.yaml",
    "../config/config.yaml",                              
    "/kaggle/input/clinical-pipeline-config/config.yaml", 
    "/kaggle/input/datasets/luwammajor/clinical-pipeline-config/config.yaml",
]

def _to_namespace(value):
    """Recursively convert nested dicts/lists into dot-accessible namespaces."""
    if isinstance(value, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_to_namespace(v) for v in value]
    return value

def load_config(config_path: str = None, env: str = None, set_seed: bool = True):
    """Loads configuration and dynamically resolves environment-specific paths."""
    if config_path is None:
        config_path = next((p for p in _DEFAULT_CANDIDATES if os.path.exists(p)), None)
        if config_path is None:
            raise FileNotFoundError(
                f"config.yaml not found in default locations: {_DEFAULT_CANDIDATES}. "
                "Provide config_path explicitly."
            )

    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    active_env = env or raw.get("active_environment", "local")
    if active_env not in raw["environments"]:
        raise KeyError(f"Environment '{active_env}' not defined in config.yaml.")

    env_paths = raw["environments"][active_env]

    resolved_paths = {
        "mimic_dir": env_paths["mimic_dir"],
        "notes_dir": env_paths["notes_dir"],
        "preprocessed_dir": env_paths["preprocessed_dir"],
        "output_dir": env_paths["output_dir"],
    }
    
    # Resolve input datasets from preprocessed_dir
    for key, rel_path in raw.get("data", {}).items():
        resolved_paths[key] = os.path.join(env_paths["preprocessed_dir"], rel_path)

    # Resolve output files from output_dir
    for key, rel_path in raw.get("outputs", {}).items():
        resolved_paths[key] = os.path.join(env_paths["output_dir"], rel_path)

    raw["paths"] = resolved_paths
    raw["active_environment"] = active_env

    cfg = _to_namespace(raw)

    if set_seed:
        random.seed(cfg.random_seed)
        np.random.seed(cfg.random_seed)

    return cfg