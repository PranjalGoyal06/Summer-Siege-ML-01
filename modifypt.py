import torch
import json
from pathlib import Path

runs = {
    "bilstm": "results/hpo_runs/bilstm",
    "esm": "results/hpo_runs/esm"
}

for model_type, dir_path in runs.items():
    pt_file = Path(dir_path) / "best_model.pt"
    json_file = Path(dir_path) / "hpo_summary.json"

    # Load checkpoint
    checkpoint = torch.load(pt_file, map_location="cpu")

    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        # If it already has "arch", just skip
        if 'arch' in checkpoint:
            print(f"{model_type}: already updated with 'arch', skipping")
            continue
        # If it has "model_arch", rename to "arch"
        if 'model_arch' in checkpoint:
            checkpoint['arch'] = checkpoint.pop('model_arch')
            torch.save(checkpoint, pt_file)
            print(f"{model_type}: renamed 'model_arch' → 'arch'")
            continue

    # If it's old-format: just state_dict
    with open(json_file, 'r') as f:
        hpo_summary = json.load(f)
    hyperparams = hpo_summary['best_trial']['hyperparameters']

    new_ckpt = {
        'state_dict': checkpoint,
        'arch': hyperparams
    }
    torch.save(new_ckpt, pt_file)
    print(f"{model_type}: upgraded checkpoint with 'arch'")
