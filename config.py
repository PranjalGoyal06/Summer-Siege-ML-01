import yaml
import torch

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    # Handle dynamic device selection
    if cfg.get("device", "auto") == "auto":
        cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    return cfg

config = load_config()
