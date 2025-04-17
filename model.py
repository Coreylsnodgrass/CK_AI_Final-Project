# model.py
import torch.nn as nn

class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=64, nhead=2, num_layers=1, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x, hidden=None):
        # x: [batch, seq_len]
        x = self.embedding(x)               # → [batch, seq_len, d_model]
        out = self.transformer(x)           # → [batch, seq_len, d_model]
        logits = self.fc(out)               # → [batch, seq_len, vocab_size]
        return logits, None