import os
import json
import ollama
from openai import OpenAI

# Configuration

SLM_MODEL = "qwen3"  # Better than phi3

BMF_API_KEY = os.getenv("BMF_API_KEY")

BMF_MODEL = "gpt-5-nano-2025-08-07"
BMF_API_VERSION = "2025-04-01-preview"

BMF_BASE_URL = (
    "https://aoai-farm.bosch-temp.com/api/openai/"
    "deployments/gpt-5-nano-2025-08-07"
)

# BMF Client

def create_bmf_client():

    if not BMF_API_KEY:
        raise ValueError(
            "BMF_API_KEY environment variable not found."
        )

    client = OpenAI(
        api_key=BMF_API_KEY,
        base_url=BMF_BASE_URL,
        default_headers={
            "genaiplatform-farm-subscription-key": BMF_API_KEY
        }
    )

    return client


# Ask Local SLM

def ask_slm(question):

    system_prompt = """
You are a lightweight local AI model.

Your task is to decide whether YOU can answer.

Rules:

1. Simple math -> can_answer=true
2. General knowledge -> can_answer=true
3. Programming questions -> can_answer=true
4. Definitions -> can_answer=true

Return can_answer=false only if:
- Latest news
- Real-time information
- Internet searches required
- Company internal information
- Information you are unsure about

Return ONLY JSON.

Example:
{
  "can_answer": true,
  "confidence": 0.95,
  "answer": "2"
}
"""

    response = ollama.chat(
        model=SLM_MODEL,
        format="json",   # IMPORTANT
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": question
            }
        ]
    )

    raw_text = response["message"]["content"]

    print("\n-------------------")
    print("RAW SLM RESPONSE")
    print(raw_text)
    print("-------------------")

    try:
        result = json.loads(raw_text)

    except Exception:

        result = {
            "can_answer": False,
            "confidence": 0.0,
            "answer": raw_text
        }

    return result


# Ask BMF

def ask_bmf_llm(client, question, conversation_history):

    messages = [
        {
            "role": "system",
            "content": "You are a helpful AI assistant."
        }
    ]

    messages.extend(conversation_history)

    messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    response = client.chat.completions.create(
        model=BMF_MODEL,
        messages=messages,
        extra_query={
            "api-version": BMF_API_VERSION
        }
    )

    return response.choices[0].message.content


# Main Chatbot


def cascaded_chatbot():

    print("=" * 60)
    print(" Cascaded Chatbot (SLM -> BMF)")
    print("=" * 60)
    print("Local SLM :", SLM_MODEL)
    print("Cloud LLM :", BMF_MODEL)
    print("Type 'exit' to quit")

    bmf_client = create_bmf_client()

    conversation_history = []

    while True:

        user_input = input("\nYou : ").strip()

        if user_input.lower() == "exit":
            break

        try:

            print("\nChecking local SLM...")

            slm_result = ask_slm(user_input)

            print("\nParsed Result:")
            print(json.dumps(slm_result, indent=2))

            can_answer = slm_result.get(
                "can_answer",
                False
            )

            confidence = float(
                slm_result.get(
                    "confidence",
                    0
                )
            )

            slm_answer = slm_result.get(
                "answer",
                ""
            )

          
            # SLM response

            if can_answer and confidence >= 0.70:

                print("\n✅ Source : Local SLM")
                print("Confidence:", confidence)

                print("\nBot:")
                print(slm_answer)

                conversation_history.append(
                    {
                        "role": "user",
                        "content": user_input
                    }
                )

                conversation_history.append(
                    {
                        "role": "assistant",
                        "content": slm_answer
                    }
                )

            # LLM fallback

            else:

                print("\n⚠ SLM not confident")
                print("Confidence:", confidence)

                print("\nSwitching to BMF...")

                llm_answer = ask_bmf_llm(
                    bmf_client,
                    user_input,
                    conversation_history
                )

                print("\n✅ Source : BMF LLM")
                print("\nBot:")
                print(llm_answer)

                conversation_history.append(
                    {
                        "role": "user",
                        "content": user_input
                    }
                )

                conversation_history.append(
                    {
                        "role": "assistant",
                        "content": llm_answer
                    }
                )

        except Exception as e:

            print("\nERROR:")
            print(e)

            print("\nCheck:")
            print("1. Ollama is running")
            print("2. qwen3 is installed")
            print("3. BMF_API_KEY is set")
            print("4. Network access is available")

    print("\nGoodbye!")



if __name__ == "__main__":
    cascaded_chatbot()