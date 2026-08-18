import ollama



def chat_with_ollama():
    """
    A simple function to create a command-line chatbot with Ollama.
    """
    print("----- Ollama Chatbot -----")
    print("Ask a question or type 'exit' to quit.")



    while True:
        # Get user input from the command line
        user_input = input("\nYou: ")



        # Check if the user wants to exit
        if user_input.lower() == 'exit':
            print("Exiting chatbot. Goodbye!")
            break



        try:
            # Send the user's message to Ollama
            response = ollama.chat(
                model='qwen3:latest',
                messages=[{'role': 'user', 'content': user_input}]
            )



            # Print the assistant's response
            print("Bot:", response['message']['content'])



        except Exception as e:
            print(f"An error occurred: {e}")
            print("Please ensure the Ollama application is running and the 'qwen3:latest' model is available.")
            break



if __name__ == "__main__":
    chat_with_ollama()