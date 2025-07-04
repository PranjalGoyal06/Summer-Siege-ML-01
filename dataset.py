import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd




class PrepareCB513(Dataset):
    def __init__(self, csv_path, window_size=17):
        self.df = pd.read_csv(csv_path)
        self.window = window_size
        self.half = window_size // 2

        self.aa_vocab = "ACDEFGHIKLMNPQRSTVWY"
        self.aa_to_idx = {aa: i for i, aa in enumerate(self.aa_vocab)}
        self.pad_idx = len(self.aa_vocab)

        self.ss_map = {'H': 0, 'E': 1, 'C': 2}
        self.samples = []
        self._prepare()

    def _prepare(self):
        for seq, ss, mask in zip(self.df['input'], self.df['dssp3'], self.df['cb513_mask']):
            seq, ss = seq.strip().upper(), ss.strip().upper()
            mask = [float(m) for m in mask.strip().split()]

            if not (len(seq) == len(ss) == len(mask)): # corrupt rows
                continue 

            padded_seq = 'X' * self.half + seq + 'X' * self.half

            for i in range(len(seq)):
                if mask[i] != 1.0:
                    continue

                window = padded_seq[i:i + self.window]
                x = [self.aa_to_idx.get(aa, self.pad_idx) for aa in window]
                y = self.ss_map[ss[i]]
                self.samples.append((x, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)




dataset = PrepareCB513("data/CB513.csv")
loader = DataLoader(dataset, batch_size=32, shuffle=True)

for xb, yb in loader:
    print(xb.shape)
    print(yb.shape)
    break