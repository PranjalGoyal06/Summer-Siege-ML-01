import torch
import torch.nn as nn

class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=50, hidden_dim=128, num_classes=3, padding_idx=20):
        super(BiLSTMClassifier, self).__init__()

        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=embedding_dim,
                                      padding_idx=padding_idx)

        self.bilstm = nn.LSTM(input_size=embedding_dim,
                              hidden_size=hidden_dim,
                              num_layers=1,
                              batch_first=True,
                              bidirectional=True)

        self.classifier = nn.Linear(in_features=2 * hidden_dim, out_features=num_classes)

    def forward(self, x):
        """
        x: (batch_size, seq_len) of integer indices
        """
        embedded = self.embedding(x)         # (batch_size, seq_len, embedding_dim)
        lstm_out, _ = self.bilstm(embedded)  # (batch_size, seq_len, 2*hidden_dim)
        center = lstm_out[:, lstm_out.shape[1] // 2, :]  # pick center position's output
        out = self.classifier(center)        # (batch_size, num_classes)
        return out
