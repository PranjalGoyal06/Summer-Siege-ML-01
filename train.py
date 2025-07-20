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
if config['use_pretrained_embeddings']:
    from datasets.dataset_esm import ESM_Embedding_Dataset as Dataset
    model_type = 'esm'
else:
    from datasets.dataset_bilstm import PrepareCB513 as Dataset
    model_type = 'bilstm'

model = BiLSTM_Model(
    hidden_dim=config[model_type]['hidden_dim'],
    dropout_rate=config[model_type]['dropout_rate'],
    use_pretrained_embeddings=config['use_pretrained_embeddings'],
    num_layers=config[model_type].get('num_layers', 1)
).to(device)

# INITIALISING DATASET
dataset = Dataset()

train_size = int(config['train_split'] * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=config[model_type]['batch_size'], shuffle=True)
val_loader = DataLoader(val_set, batch_size=config[model_type]['batch_size'])

loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config[model_type]['learning_rate'])

# FUNCTIONS
def train_one_epoch(model, loader, optimizer, loss_func, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        output = model(xb)
        loss = loss_func(output, yb)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * xb.size(0)
        _, predicted = torch.max(output.data, 1)
        total += yb.size(0)
        correct += (predicted == yb).sum().item()
    
    return total_loss / len(loader.dataset), correct / total

def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            output = model(xb)
            loss = loss_func(output, yb)
            
            total_loss += loss.item() * xb.size(0)
            _, predicted = torch.max(output.data, 1)
            total += yb.size(0)
            correct += (predicted == yb).sum().item()
    
    return correct / total



if __name__ == "__main__":
    print(f"Training {model_type} with {config['train_split'] * 100}% of the dataset...")

    # TRAINING LOOP
    best_val_acc = 0
    epochs_no_improve = 0
    start_time = time.time()

    # Setup checkpoint path
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"{model_type}_{run_timestamp}.pt")

    for epoch in range(config[model_type]['epochs']):
        epoch_start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, loss_func, device)
        val_acc = evaluate(model, val_loader, device)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), checkpoint_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        # Early stopping
        if epochs_no_improve >= config[model_type]['patience']:
            print(f'\nEarly stopping after {epoch + 1} epochs')
            break
        
        epoch_time = time.time() - epoch_start
        print(f'Epoch {epoch + 1}/{config[model_type]["epochs"]} - Train Loss: {train_loss:.4f}, ' 
              f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Time: {epoch_time:.1f}s')

    print(f'\nTraining complete. Best validation accuracy: {best_val_acc:.4f}')
    print(f'Total training time: {(time.time() - start_time) / 60:.1f} minutes')