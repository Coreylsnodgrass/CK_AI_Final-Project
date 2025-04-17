import warnings
warnings.filterwarnings("ignore", message="enable_nested_tensor is True")

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from multiprocessing import freeze_support
from model import TinyTransformer

# ----------------------------
# Inline tokenizer (whitespace‑level)
# ----------------------------
with open("data.txt", "r", encoding="utf-8", errors="ignore") as f:
    tokens = f.read().split()

vocab      = sorted(set(tokens))
token2idx  = {t: i for i, t in enumerate(vocab)}
vocab_size = len(vocab)
encoded    = torch.tensor([token2idx[t] for t in tokens], dtype=torch.long)
# ----------------------------

# ========== CONFIGURATION ==========
SEQ_LENGTH      = 16      # shorter context to reduce memory
BATCH_SIZE      = 64      # smaller batch to stay within 1 GB RAM
D_MODEL         = 32      # increased model dimensionality for better capacity
NHEAD           = 1       # single attention head
DIM_FEEDFORWARD = 64      # increased feedforward layer size for expressivity
EPOCHS          = 5       # number of epochs
LEARNING_RATE   = 0.003
DEVICE          = torch.device("cpu")  # CPU-only
# ====================================

# Precompute sliding windows: [num_sequences, SEQ_LENGTH+1]
sequences = encoded.unfold(0, SEQ_LENGTH + 1, 1)
# Split into inputs and targets
x_data = sequences[:, :-1].contiguous()
y_data = sequences[:, 1:].contiguous()

dataset = TensorDataset(x_data, y_data)  # pre-batched dataset


def train():
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )

    model = TinyTransformer(
        vocab_size=vocab_size,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=2,           # upgraded to 2 layers for more capacity
        dim_feedforward=DIM_FEEDFORWARD
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
        for x, y in tqdm(dataloader, desc="Batches", unit="batch"):
            x, y = x.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()
            logits, _ = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), "tiny_transformer.pth")
    print("Model saved as tiny_transformer.pth")

if __name__ == "__main__":
    freeze_support()
    train()
