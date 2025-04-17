# interface.py

from generate import generate_text

def main():
    print("Welcome to the Tiny LLM Demo!")
    while True:
        prompt = input("\nEnter a prompt (or 'quit' to exit): ")
        if prompt.lower() == "quit":
            break
        output = generate_text(prompt, length=50)
        print("\n⟶", output)

if __name__ == "__main__":
    main()
