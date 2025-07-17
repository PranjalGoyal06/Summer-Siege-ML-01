import torch
import torch.nn as nn
from config import config

class BiLSTM_Model(nn.Module):
    def __init__(self, vocab_size = 21, embedding_dim=config['embedding_dim'], hidden_dim=config['hidden_dim'],
                    num_classes=3, padding_idx=20, dropout_rate = config['dropout_rate']):

        super(BiLSTM_Model, self).__init__()

        self.embedding = nn.Embedding(num_embeddings = vocab_size,
                                      embedding_dim  = embedding_dim,
                                      padding_idx    = padding_idx)

        self.bilstm = nn.LSTM(input_size    = embedding_dim,
                              hidden_size   = hidden_dim,
                              num_layers    = 1,
                              batch_first   = True,
                              bidirectional = True)

        self.classifier = nn.Linear(in_features = 2*hidden_dim, out_features = num_classes)
        
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):

        embedded = self.embedding(x) 
        lstm_out, _ = self.bilstm(embedded)
        center = lstm_out[:, lstm_out.shape[1] // 2, :] 
        dropped = self.dropout(center)
        out = self.classifier(dropped) 
        
        return out