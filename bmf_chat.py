import os
from openai import OpenAI


# BMF configuration
BMF_API_KEY = os.getenv("BMF_API_KEY")

BASE_URL = "https://aoai-farm.bosch-temp.com/api/openai/deployments/gpt-5-nano-2025-08-07"
MODEL_NAME = "gpt-5-nano-2025-08-07"
API_VERSION = "2025-04-01-preview"


def create_bmf_client():
    """
    Create BMF client using OpenAI-compatible SDK.
    """

    if not BMF_API_KEY:
        raise ValueError(
            "BMF_API_KEY not found. Please set the BMF_API_KEY environment variable."
        )

    client = OpenAI(
        api_key=BMF_API_KEY,
        base_url=BASE_URL,

        # This extra header is added because some BMF docs/welcome emails mention
        # a custom subscription-key header. It is safe to send along with Bearer auth.
        default_headers={
            "genaiplatform-farm-subscription-key": BMF_API_KEY
        }
    )

    return client


def chat_with_bmf():
    """
    Simple command-line chatbot using Bosch Model Farm.
    """

    client = create_bmf_client()

    print("----- BMF Chatbot -----")
    print("Ask a question or type 'exit' to quit.")

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        }
    ]

    while True:
        user_input = input("\nYou: ")

        if user_input.lower() == "exit":
            print("Exiting chatbot. Goodbye!")
            break

        messages.append(
            {
                "role": "user",
                "content": user_input
            }
        )

        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                extra_query={
                    "api-version": API_VERSION
                }
            )

            bot_reply = response.choices[0].message.content

            print("\nBot:", bot_reply)

            messages.append(
                {
                    "role": "assistant",
                    "content": bot_reply
                }
            )

        except Exception as e:
            print("\nAn error occurred:")
            print(e)
            print("\nPlease check:")
            print("1. BMF_API_KEY is set correctly")
            print("2. You are connected to Bosch network/VPN if required")
            print("3. The model/deployment name is available for your subscription")
            print("4. The BMF endpoint is reachable")
            break


if __name__ == "__main__":
    chat_with_bmf()