# generate.py
import torch
from model import TinyTransformer

# ————————————
# Rebuild the whitespace tokenizer
# ————————————
with open("data.txt", "r", encoding="utf-8", errors="ignore") as f:
    tokens = f.read().split()
vocab = sorted(set(tokens))
token2idx = {t: i for i, t in enumerate(vocab)}
idx2token = {i: t for i, t in enumerate(vocab)}

# ————————————
# Instantiate and load your 2‑layer Transformer
# ————————————
model = TinyTransformer(
    vocab_size=len(vocab),
    d_model=32,
    nhead=1,
    num_layers=2,
    dim_feedforward=64
)
model.load_state_dict(torch.load("tiny_transformer.pth", map_location="cpu"))
model.eval()

def generate_text(prompt, length=50, temperature=0.8, top_k=50):
    # tokenize prompt
    words = prompt.split()
    idxs  = [token2idx.get(w, 0) for w in words]
    input_ids = torch.tensor([idxs], dtype=torch.long)
    generated = words.copy()

    # track last 3‑grams
    seen_3grams = set()

    with torch.no_grad():
        for _ in range(length):
            logits, _ = model(input_ids)
            last_logits = logits[:, -1, :]                # [1, V]
            scaled = last_logits / temperature

            # top-k filtering
            if top_k is not None:
                v, idx = torch.topk(scaled, top_k)
                mask = torch.full_like(scaled, -float("Inf"))
                mask.scatter_(1, idx, v)
                scaled = mask

            probs = torch.nn.functional.softmax(scaled, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).item()

            # build candidate 3‑gram
            cand = tuple(generated[-2:] + [idx2token[next_id]])  # last 2 words + new one
            if cand in seen_3grams:
                # ban this token next time
                scaled[0, next_id] = -float("Inf")
                probs = torch.nn.functional.softmax(scaled, dim=-1)
                next_id = torch.multinomial(probs, 1).item()
            else:
                seen_3grams.add(cand)

            # append and shift
            generated.append(idx2token[next_id])
            input_ids = torch.cat([input_ids, torch.tensor([[next_id]] )], dim=1)
            # optionally: input_ids = input_ids[:, -SEQ_LENGTH:]  # sliding window

    return " ".join(generated)


if __name__ == "__main__":
    # simple CLI test
    p = input("Prompt: ")
    print("\n" + generate_text(p))
