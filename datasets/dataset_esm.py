import os
import torch
from torch.utils.data import Dataset
import pandas as pd
from config import config

class ESM_Embedding_Dataset(Dataset):
    def __init__(self, csv_path=config["dataset_path"], embeddings_dir=config["esm_embeddings_path"], window_size=config["window_size"]):
        self.window_size = window_size
        self.half_window = window_size // 2
        self.embeddings_dir = embeddings_dir

        df = pd.read_csv(csv_path)
        
        self.sequences = []
        for i, row in df.iterrows():
            seq_id = f"seq_{i:03}" 
            sequence = list(row["input"])
            labels = list(row["dssp3"])
            mask = [float(x) for x in str(row["cb513_mask"]).split()]

            self.sequences.append((seq_id, sequence, labels, mask))

        # one training example
        self.samples = []
        for seq_id, _, labels, mask in self.sequences:
            for i in range(len(labels)):
                if mask[i] == 1.0:
                    self.samples.append((seq_id, i, labels[i]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq_id, center_idx, label = self.samples[idx]

        embedding_path = os.path.join(self.embeddings_dir, f"{seq_id}.pt")
        full_embedding = torch.load(embedding_path, weights_only=True)  # shape: [L, 320], L = length of protein seq

        # edge handling
        pad_dim = full_embedding.shape[1]
        padded = torch.zeros((len(full_embedding) + 2 * self.half_window, pad_dim))
        padded[self.half_window : self.half_window + len(full_embedding)] = full_embedding

        # extract centered window
        start = center_idx
        end = center_idx + self.window_size
        window = padded[start:end]

        self.ss_map = {'H': 0, 'E': 1, 'C': 2}
        label_idx = self.ss_map[label]

        return window.clone().detach().float(), torch.tensor(label_idx).clone().detach().long()



# ds = ESM_Embedding_Dataset()

# print(ds.__len__())
