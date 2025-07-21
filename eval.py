import os
import argparse
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from models.bilstm import BiLSTM_Model
from config import config

device = config["device"]

def get_checkpoint_path(ckpt_arg):
    if ckpt_arg:
        if not os.path.isfile(ckpt_arg):
            raise FileNotFoundError(f"Checkpoint '{ckpt_arg}' does not exist.")
        return ckpt_arg

    model_type = "esm" if config["use_pretrained_embeddings"] else "bilstm"
    ckpts = [os.path.join("checkpoints", f) for f in os.listdir("checkpoints")
             if f.startswith(model_type) and f.endswith(".pt")]
    return max(ckpts, key=os.path.getmtime)

def get_model_type(ckpt_path):
    name = os.path.basename(ckpt_path)
    if name.startswith("esm"):
        config["use_pretrained_embeddings"] = True
        return "esm"
    if name.startswith("bilstm"):
        config["use_pretrained_embeddings"] = False
        return "bilstm"
    return "esm" if config["use_pretrained_embeddings"] else "bilstm"

def load_checkpoint(ckpt_path, model_type):
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"], checkpoint.get("arch", config[model_type]) # matching hyperparams for that checkpoint
    if isinstance(checkpoint, dict):
        return checkpoint, config[model_type] # present config, may/may not match
    raise ValueError("Unknown checkpoint format")

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint.")
    parser.add_argument("--ckpt", type=str, help="Path to checkpoint file.")
    args = parser.parse_args()

    ckpt_path = get_checkpoint_path(args.ckpt)
    model_type = get_model_type(ckpt_path)
    print(f"Loading checkpoint: {os.path.basename(ckpt_path)}")

    state_dict, arch = load_checkpoint(ckpt_path, model_type)

    model = BiLSTM_Model(
        hidden_dim=arch["hidden_dim"],
        num_layers=arch["num_layers"],
        dropout_rate=arch["dropout_rate"],
        use_pretrained_embeddings=config["use_pretrained_embeddings"],
        embedding_dim=arch.get("embedding_dim", 16)
    ).to(device)

    missing, unexpected = model.load_state_dict(state_dict, strict=False) # likely for older checkpoints
    if missing: print(f"Warning: Missing keys: {missing}")
    if unexpected: print(f"Warning: Unexpected keys: {unexpected}")

    model.eval()

    if model_type == "esm":
        import datasets.dataset_esm
        dataset = datasets.dataset_esm.ESM_Embedding_Dataset()
    else:
        import datasets.dataset_bilstm
        dataset = datasets.dataset_bilstm.PrepareCB513()

    loader = DataLoader(dataset, batch_size=config[model_type]["batch_size"], shuffle=False)

    true_labels, pred_labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb).argmax(1)
            true_labels.extend(yb.cpu().numpy())
            pred_labels.extend(preds.cpu().numpy())

    print(f"\nTest Accuracy: {accuracy_score(true_labels, pred_labels):.4f}")
    print("\nClassification Report:")
    print(classification_report(true_labels, pred_labels, target_names=["H", "E", "C"]))
    print("\nConfusion Matrix:")
    print(confusion_matrix(true_labels, pred_labels))

if __name__ == "__main__":
    main()
