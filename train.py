import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from tokenizer import char2idx, idx2char, encoded_text
from model import TinyRNN

# ========== CONFIGURATION ==========
SEQ_LENGTH = 100
BATCH_SIZE = 64
EMBEDDING_DIM = 64
HIDDEN_DIM = 128
EPOCHS = 10
LEARNING_RATE = 0.003
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== DATASET ==========
class JokeDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx:idx+self.seq_length], dtype=torch.long)
        y = torch.tensor(self.data[idx+1:idx+self.seq_length+1], dtype=torch.long)
        return x, y

# ========== DATA LOADER ==========
dataset = JokeDataset(encoded_text, SEQ_LENGTH)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ========== MODEL ==========
model = TinyRNN(vocab_size=len(char2idx), embed_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ========== TRAIN LOOP ==========
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for x, y in dataloader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        output, _ = model(x)
        loss = criterion(output.view(-1, output.size(-1)), y.view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}")

# ========== SAVE MODEL ==========
torch.save(model.state_dict(), "tiny_rnn.pth")
print("Model saved as tiny_rnn.pth")