# generate.py

import torch
from model import TinyTransformer

# ————————————
# Inline tokenizer (whitespace‑level)
# ————————————
with open("data.txt", "r", encoding="utf-8", errors="ignore") as f:
    tokens = f.read().split()

vocab      = sorted(set(tokens))
token2idx  = {t: i for i, t in enumerate(vocab)}
idx2token  = {i: t for i, t in enumerate(vocab)}
vocab_size = len(vocab)

# ————————————
# Instantiate Transformer
# ————————————
# (must match your train.py hyperparams)
model = TinyTransformer(
    vocab_size=vocab_size,
    d_model=16,
    nhead=1,
    num_layers=1,
    dim_feedforward=32
)
model.load_state_dict(torch.load("tiny_transformer.pth", map_location="cpu"))
model.eval()

def generate_text(prompt, length=50):
    # encode prompt as whitespace tokens
    words = prompt.split()
    indices = [token2idx.get(w, 0) for w in words]
    inp = torch.tensor([indices], dtype=torch.long)

    out = words.copy()
    with torch.no_grad():
        hidden = None
        for _ in range(length):
            logits, hidden = model(inp, hidden)
            last_logits = logits[:, -1, :]   # [1, vocab_size]
            probs = torch.nn.functional.softmax(last_logits, dim=-1)
            idx = torch.multinomial(probs, num_samples=1).item()
            out.append(idx2token[idx])
            inp = torch.tensor([[idx]], dtype=torch.long)

    return " ".join(out)

if __name__ == "__main__":
    prompt = input("Prompt: ")
    print(generate_text(prompt))
