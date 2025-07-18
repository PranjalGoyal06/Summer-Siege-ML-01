import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import time
import os
from datetime import datetime
from models.bilstm import BiLSTM_Model
from config import config

device = config['device']
print("Using CUDA/CPU: ", device)

# CHOOSING MODEL
if config["use_pretrained_embeddings"] == False:
    from datasets.dataset_bilstm import PrepareCB513 as Dataset
    model = BiLSTM_Model(use_pretrained_embeddings=False).to(device)
else:
    from datasets.dataset_esm import ESM_Embedding_Dataset as Dataset
    model = BiLSTM_Model(use_pretrained_embeddings=True).to(device)

# Setup checkpoint path
run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)
model_type = "bilstm" if not config["use_pretrained_embeddings"] else "esm" 
checkpoint_path = os.path.join(checkpoint_dir, f"{model_type}_{run_timestamp}.pt")

# INITIALISING DATASET
dataset = Dataset()

train_size = int(config['train_split'] * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True)
val_loader = DataLoader(val_set, batch_size=config['batch_size'])

loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

# -----------TRAINING----------- #

def train_one_epoch(model, loader):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = loss_func(out, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * yb.size(0) 
        _, preds = torch.max(out, dim=1)
        correct += (preds == yb).sum().item()
        total += yb.size(0)

    avg_loss = total_loss / total
    acc = correct / total
    return avg_loss, acc

def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            _, preds = torch.max(out, dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
    return correct / total

# TRAINING LOOP
best_val_acc = 0
epochs_no_improve = 0
start_time = time.time()

for epoch in range(config['epochs']):
    epoch_start = time.time()

    train_loss, train_acc = train_one_epoch(model, train_loader)
    val_acc = evaluate(model, val_loader)

    print(f"[Epoch {epoch+1}] Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Time: {time.time() - epoch_start:.2f}s")

    # Save model if validation improves
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), checkpoint_path)

        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        print(f"⚠️ No improvement for {epochs_no_improve} epoch(s)")
        if epochs_no_improve >= config['patience']:
            print("⛔ Early stopping triggered.")
            break

total_time = time.time() - start_time
print(f"\nTraining complete in {total_time:.2f} seconds. Best Val Acc: {best_val_acc:.4f}")
print(f"Best model checkpoint saved to {checkpoint_path}")
