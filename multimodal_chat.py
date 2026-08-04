import ollama
from pathlib import Path

MODEL_NAME = "llama3.2-vision"

messages = []

def ask_model(user_text, image_path=None):

    message = {
        "role": "user",
        "content": user_text
    }

    if image_path:

        image_file = Path(image_path)

        if not image_file.exists():
            return f"Image not found: {image_path}"

        message["images"] = [str(image_file)]

    messages.append(message)

    response = ollama.chat(
        model=MODEL_NAME,
        messages=messages
    )

    answer = response["message"]["content"]

    messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

    return answer


print("===== Multimodal Chatbot =====")
# print("Model:", MODEL_NAME)
print("Type 'exit' to quit")

while True:

    question = input("\nYou: ")

    if question.lower() == "exit":
        break

    image_path = input(
        "Image path (Enter to skip): "
    ).strip()

    if image_path == "":
        image_path = None

    try:
        answer = ask_model(
            question,
            image_path
        )

        print("\nBot:", answer)

    except Exception as e:
        print("Error:", e)