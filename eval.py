import torch
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from models.bilstm import BiLSTM_Model
from config import config
import os

device = config["device"]

if not config["use_pretrained_embeddings"]:
    from datasets.dataset_bilstm import PrepareCB513 as Dataset
    model = BiLSTM_Model(use_pretrained_embeddings=False).to(device)
    model_type = 'bilstm'
else:
    from datasets.dataset_esm import ESM_Embedding_Dataset as Dataset
    model = BiLSTM_Model(use_pretrained_embeddings=True).to(device)
    model_type = 'esm'

# Load latest checkpoint
checkpoint_dir = "checkpoints"
matching_ckpts = [
    os.path.join(checkpoint_dir, f)
    for f in os.listdir(checkpoint_dir)
    if f.startswith(model_type) and f.endswith(".pt")
]
if not matching_ckpts:
    raise FileNotFoundError(f"No checkpoints found for model type '{model_type}' in '{checkpoint_dir}'.")

latest_ckpt_path = max(matching_ckpts, key=os.path.getmtime)
print(f"Loading latest {model_type} checkpoint: {latest_ckpt_path}")

state_dict = torch.load(latest_ckpt_path, map_location=device)
model.load_state_dict(state_dict, strict=False)
model.eval()

dataset = Dataset()
loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False)

# Evaluate
all_preds, all_labels = [], []

with torch.no_grad():
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        out = model(xb)
        _, preds = torch.max(out, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(yb.cpu().numpy())

# Metrics
print("Classification Report:")
report_str = classification_report(all_labels, all_preds, target_names=["H", "E", "C"])
print(report_str)
report_dict = classification_report(all_labels, all_preds, target_names=["H", "E", "C"], output_dict=True)
df = pd.DataFrame(report_dict).transpose()
df.to_csv(os.path.join("results", f"{model_type}_report.csv"))

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["H", "E", "C"],
            yticklabels=["H", "E", "C"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join("results", f"{model_type}_confusion_matrix.png"))
plt.close()

# Metrics
print("Classification Report:")
report_str = classification_report(all_labels, all_preds, target_names=["H", "E", "C"])
print(report_str)
report_dict = classification_report(all_labels, all_preds, target_names=["H", "E", "C"], output_dict=True)
df = pd.DataFrame(report_dict).transpose()
df.to_csv(os.path.join("results", f"{model_type}_report.csv"))

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["H", "E", "C"],
            yticklabels=["H", "E", "C"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join("results", f"{model_type}_confusion_matrix.png"))
plt.close()