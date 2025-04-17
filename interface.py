# interface.py
from generate import generate_text

def main():
    print("Welcome to TinyLLM!")
    while True:
        prompt = input("\nEnter a prompt (or 'quit' to exit): ")
        if prompt.strip().lower() == "quit":
            break
        print("\n⟶", generate_text(prompt, length=50))

if __name__ == "__main__":
    main()
