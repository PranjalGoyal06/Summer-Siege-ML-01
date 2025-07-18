import torch
import torch.nn as nn
from config import config

class BiLSTM_Model(nn.Module):
    def __init__(self,
                 vocab_size=21,
                 embedding_dim=config['embedding_dim'],
                 hidden_dim=config['hidden_dim'],
                 num_classes=3,
                 padding_idx=20,
                 dropout_rate=config['dropout_rate'],
                 use_pretrained_embeddings=config['use_pretrained_embeddings']):

        super(BiLSTM_Model, self).__init__()
        self.use_pretrained_embeddings = use_pretrained_embeddings

        if not self.use_pretrained_embeddings:
            self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                          embedding_dim=embedding_dim,
                                          padding_idx=padding_idx)
            lstm_input_dim = embedding_dim
        else:
            lstm_input_dim = 320  # fixed for ESM

        self.bilstm = nn.LSTM(input_size=lstm_input_dim,
                              hidden_size=hidden_dim,
                              num_layers=1,
                              batch_first=True,
                              bidirectional=True)

        self.classifier = nn.Linear(in_features=2 * hidden_dim, out_features=num_classes)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        """
        x: 
        - if use_pretrained_embeddings=False → tensor of token indices [batch_size, window_size]
        - if use_pretrained_embeddings=True → tensor of embeddings [batch_size, window_size, 320]
        """

        if not self.use_pretrained_embeddings:
            x = self.embedding(x)  # [batch_size, window_size, embedding_dim]

        lstm_out, _ = self.bilstm(x)
        center = lstm_out[:, lstm_out.shape[1] // 2, :]
        dropped = self.dropout(center)
        out = self.classifier(dropped)

        return out
