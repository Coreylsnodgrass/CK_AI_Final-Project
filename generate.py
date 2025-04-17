import torch
from model import TinyRNN
from tokenizer import encoded_text, char2idx, idx2char, vocab_size

model = TinyRNN(vocab_size)
model.load_state_dict(torch.load("tiny_rnn.pth"))
model.eval()

def generate_text(prompt, length=200):
    input_indices = torch.tensor([[char2idx[c] for c in prompt]], dtype=torch.long)
    hidden = None
    output_text = prompt

    with torch.no_grad():
        for _ in range(length):
            output, hidden = model(input_indices, hidden)
            last_logits = output[:, -1, :]
            prob = torch.nn.functional.softmax(last_logits, dim=-1)
            next_idx = torch.multinomial(prob, num_samples=1).item()
            next_char = idx2char[next_idx]
            output_text += next_char
            input_indices = torch.tensor([[next_idx]], dtype=torch.long)

    return output_text

if __name__ == "__main__":
    prompt = input("Prompt: ")
    print(generate_text(prompt))
