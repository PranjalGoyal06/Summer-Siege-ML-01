import torch
import torch.nn as nn
from config import config

model_type = "esm" if config['use_pretrained_embeddings'] else "bilstm"

class BiLSTM_Model(nn.Module):
    def __init__(self,
                 vocab_size=21,
                 embedding_dim=config['bilstm']['embedding_dim'],
                 hidden_dim=config[model_type]['hidden_dim'],
                 num_classes=3,
                 padding_idx=20,
                 num_layers=config[model_type]['num_layers'],
                 dropout_rate=config[model_type]['dropout_rate'],
                 use_pretrained_embeddings=config['use_pretrained_embeddings'],
                 ):

        super(BiLSTM_Model, self).__init__()
        self.use_pretrained_embeddings = use_pretrained_embeddings

        # Embedding layer
        if not self.use_pretrained_embeddings:
            self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                          embedding_dim=embedding_dim,
                                          padding_idx=padding_idx)
            lstm_input_dim = embedding_dim
        else:
            lstm_input_dim = 320  # fixed for ESM

        # BiLSTM layer
        self.bilstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )

        # Output layer
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        
        if not self.use_pretrained_embeddings:
            x = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        
        lstm_out, _ = self.bilstm(x)  # [batch_size, seq_len, hidden_dim * 2]
        
        # Use the last hidden state for classification
        # Taking the mean of the sequence dimension
        out = lstm_out.mean(dim=1)  # [batch_size, hidden_dim * 2]
        
        out = self.dropout(out)
        out = self.fc(out)  # [batch_size, num_classes]
        
        return out
