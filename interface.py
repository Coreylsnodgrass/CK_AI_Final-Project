from generate import generate_text

def main():
    print("Welcome to the Tiny Dad Joke Generator!")
    while True:
        prompt = input("Enter a prompt (or 'quit' to exit): ")
        if prompt.lower() == 'quit':
            break
        result = generate_text(prompt, length=200)
        print("\nGenerated Joke:")
        print(result)
        print("\n")

if __name__ == "__main__":
    main()
