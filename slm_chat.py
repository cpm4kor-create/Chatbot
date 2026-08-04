import ollama

def chat():
    print("===== SLM Chatbot (Phi-3) =====")
    print("Type 'exit' to quit")

    while True:
        user_input = input("\nYou: ")

        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        try:
            response = ollama.chat(
                model="phi3",
                messages=[
                    {
                        "role": "user",
                        "content": user_input
                    }
                ]
            )

            print("\nBot:", response["message"]["content"])

        except Exception as e:
            print("Error:", e)

if __name__ == "__main__":
    chat()