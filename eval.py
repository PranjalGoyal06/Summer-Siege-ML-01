import torch
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from torch.utils.data import DataLoader
from models.bilstm import BiLSTM_Model
from config import config
import os

device = config["device"]

# Choosing Dataset and model
if config['use_pretrained_embeddings']:
    model_type = 'esm'
else:
    model_type = 'bilstm'

if model_type == 'esm':
    from datasets.dataset_esm import ESM_Embedding_Dataset as Dataset
else:
    from datasets.dataset_bilstm import PrepareCB513 as Dataset

# Initialize model with config parameters
model = BiLSTM_Model(
    hidden_dim=config[model_type]['hidden_dim'],
    dropout_rate=config[model_type]['dropout_rate'],
    use_pretrained_embeddings=config['use_pretrained_embeddings'],
    num_layers=config[model_type].get('num_layers', 1)  
).to(device)

# Load checkpoint
checkpoint_dir = "checkpoints"
matching_ckpts = [
    os.path.join(checkpoint_dir, f)
    for f in os.listdir(checkpoint_dir)
    if f.startswith(model_type) and f.endswith(".pt")
]
if not matching_ckpts:
    raise FileNotFoundError(f"No checkpoints found for '{model_type}' in '{checkpoint_dir}'.")

ckpt_path = max(matching_ckpts, key=os.path.getmtime)
print(f"Attempting to load checkpoint: {os.path.basename(ckpt_path)}")
model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
model.eval()
print(f"Model loaded successfully with architecture: hidden_dim={model.bilstm.hidden_size}, num_layers={model.bilstm.num_layers}")


# Load and prepare test data
dataset = Dataset()
test_loader = DataLoader(dataset, batch_size=config[model_type]['batch_size'], shuffle=False)

# Evaluation
true_labels = []
pred_labels = []

with torch.no_grad():
    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)
        output = model(xb)
        _, preds = torch.max(output, 1)
        
        true_labels.extend(yb.cpu().numpy())
        pred_labels.extend(preds.cpu().numpy())

# Calculate metrics
accuracy = accuracy_score(true_labels, pred_labels)
print(f"\nTest Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(true_labels, pred_labels, target_names=["H", "E", "C"]))
print("\nConfusion Matrix:")
print(confusion_matrix(true_labels, pred_labels))