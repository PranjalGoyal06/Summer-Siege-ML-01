import torch
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from datasets.cb513_dataset import PrepareCB513
from config import config
from models.bilstm import BiLSTM_Model

def evaluate(model, loader, device):

    '''
    -> computes basic accuracy (correct/total)
    -> Q3 classification report
    -> confusion matrix
    '''

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            _, preds = torch.max(out, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(yb.cpu().numpy())

    # Basic accuracy
    correct = np.sum(np.array(all_preds) == np.array(all_labels))
    acc = correct / len(all_labels)

    # Print classification report
    print("\n📊 Classification Report (Q3):")
    print(classification_report(all_labels, all_preds, target_names=["H", "E", "C"]))

    # Plot confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["H", "E", "C"], yticklabels=["H", "E", "C"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

    return acc

dataset = PrepareCB513()
train_size = int(config['train_split'] * len(dataset))
val_size = len(dataset) - train_size
_, val_set = random_split(dataset, [train_size, val_size])

val_loader = DataLoader(val_set, batch_size=config['batch_size'])

model = BiLSTM_Model().to(config['device'])
model.load_state_dict(torch.load("checkpoints/best_model.pt"))

evaluate(model, val_loader, config['device'])
