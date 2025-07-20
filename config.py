import yaml
import torch
from pathlib import Path

def load_config(path=None):
    if path is None:
        path = Path(__file__).parent / "config.yaml"
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if cfg.get("device", "auto") == "auto":
        cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    return cfg

config = load_config()

