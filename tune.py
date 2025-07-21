import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import os
import json
import random
import itertools
import time
from datetime import datetime
from models.bilstm import BiLSTM_Model
from config import load_config
from train import train_one_epoch, evaluate

def train_model(model_type, hyperparams, base_config):
    start_time = time.time()
    config = base_config.copy()
    config.update(hyperparams)
    device = config['device']
    epochs = config[model_type]['epochs']  # Get epochs from config based on model type
    
    if model_type == 'esm':
        from datasets.dataset_esm import ESM_Embedding_Dataset as Dataset
    else:
        from datasets.dataset_bilstm import PrepareCB513 as Dataset
    
    model = BiLSTM_Model(
        hidden_dim=config[model_type]['hidden_dim'],
        dropout_rate=config[model_type]['dropout_rate'],
        use_pretrained_embeddings=config['use_pretrained_embeddings'],
        num_layers=config[model_type].get('num_layers', 1)
    ).to(device)
    
    # Dataset setup
    dataset = Dataset()
    
    indices = list(range(len(dataset)))
    labels = [dataset[i][1].item() for i in indices]
    train_idx, val_idx = train_test_split(indices, test_size=1-config['train_split'], stratify=labels, random_state=42)
    train_loader = DataLoader(torch.utils.data.Subset(dataset, train_idx), 
                            batch_size=config[model_type]['batch_size'], shuffle=True)
    val_loader = DataLoader(torch.utils.data.Subset(dataset, val_idx), 
                          batch_size=config[model_type]['batch_size'])
    
    # Training
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config[model_type]['learning_rate'])
    
    best_val_acc = 0
    best_model_state = None
    epochs_no_improve = 0
    total_epochs = 0
    
    for epoch in range(epochs):
        total_epochs = epoch + 1
        train_one_epoch(model, train_loader, optimizer, loss_func, device)
        val_acc = evaluate(model, val_loader, device)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= config[model_type]['patience']:
                break
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    trial_time = time.time() - start_time
    return best_val_acc, best_model_state, total_epochs, trial_time

def generate_param_combinations(hpo_config):
    search_method = hpo_config.get('search_method', 'random')
    n_trials = hpo_config.get('n_trials', 30)
    
    param_ranges = {k: v for k, v in hpo_config.items() if k not in ['search_method', 'n_trials']}
    
    if search_method == 'grid':
        return [dict(zip(param_ranges.keys(), values)) 
               for values in itertools.product(*param_ranges.values())]
    return [dict(zip(param_ranges.keys(), 
                    [random.choice(param_ranges[k]) for k in param_ranges]))
           for _ in range(n_trials)]

config = load_config()

# Determine model type based on use_pretrained_embeddings
if config['use_pretrained_embeddings']:
    model_type = 'esm'
    print("\nTuning BiLSTM with pre-trained ESM embeddings...")
else:
    model_type = 'bilstm'
    print("\nTuning BiLSTM with learned embeddings...")

session_dir = f"results/hpo_runs/{model_type}"
os.makedirs(session_dir, exist_ok=True)

hpo_config = config['hpo'][model_type]
param_combinations = generate_param_combinations(hpo_config)
print(f"Running {len(param_combinations)} trials with {hpo_config['search_method']} search")

all_trials = []
best_trial = None
best_model_state = None

for i, hyperparams in enumerate(param_combinations, 1):
    print(f"\nTrial {i}/{len(param_combinations)}: {hyperparams}")
    
    try:
        start_time = time.time()
        val_acc, model_state, epochs_trained, trial_time = train_model(model_type, hyperparams, config)
        trial_result = {
            "trial": i, 
            "params": hyperparams, 
            "val_accuracy": val_acc,
            "epochs_trained": epochs_trained,
            "time_taken": trial_time
        }
        all_trials.append(trial_result)
        
        if not best_trial or val_acc > best_trial["val_accuracy"]:
            best_trial = {
                "hyperparameters": hyperparams, 
                "val_accuracy": val_acc,
                "epochs_trained": epochs_trained,
                "time_taken": trial_time
            }
            best_model_state = model_state
        
        print(f"  Epochs: {epochs_trained}  |  Time: {trial_time:.1f}s  |  Val Acc: {val_acc:.4f}")
    
    except Exception as e:
        print(f"Failed: {e}")
        all_trials.append({"trial": i, "params": hyperparams, 
                         "val_accuracy": 0.0, "error": str(e)})

# Save results
with open(f"{session_dir}/hpo_summary.json", 'w') as f:
    json.dump({
        "timestamp": datetime.now().isoformat(),
        "search_method": hpo_config['search_method'],
        "total_trials": len(param_combinations),
        "best_trial": best_trial,
        "all_trials": all_trials
    }, f, indent=2)

if best_model_state and best_trial:
    # Save model with architecture parameters
    checkpoint = {
        'state_dict': best_model_state,
        'model_arch': best_trial['hyperparameters'],
    }
    torch.save(checkpoint, f"{session_dir}/best_model.pt")

if best_trial:
    print(f"BEST {model_type}: {best_trial['val_accuracy']:.4f} with {best_trial['hyperparameters']}")

print(f"Results saved to {session_dir}")
print("\nHPO Complete!")