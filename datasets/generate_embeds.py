import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from config import config

os.makedirs(config['esm_embeddings_path'], exist_ok=True)

# Load ESM-2 model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(config["esm_model"])
model = AutoModel.from_pretrained(config["esm_model"])
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Load dataset
df = pd.read_csv(config["dataset_path"])

print(f"Generating ESM embeddings for {len(df)} sequences...")

for idx, row in tqdm(df.iterrows(), total=len(df)):
    seq = row["input"]
    uid = f"seq_{idx:03}"  # Unique ID for saving

    inputs = tokenizer(seq, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        emb = outputs.last_hidden_state.squeeze(0)  # (seq_len+2, emb_dim)

        # Strip [CLS] and [EOS] tokens → retain only amino acids
        emb = emb[1:-1]  # shape: (seq_len, embedding_dim)

    # Save as .pt file
    torch.save(emb.cpu(), os.path.join(config["esm_embeddings_path"], f"{uid}.pt"))
