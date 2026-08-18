import ollama

MODEL_NAME = "phi3"


def load_knowledge():
    with open("data/knowledge.txt", "r", encoding="utf-8") as f:
        return f.read()


def search_context(question, knowledge):

    question_words = question.lower().split()

    relevant_lines = []

    for line in knowledge.split("\n"):

        for word in question_words:

            if word in line.lower():
                relevant_lines.append(line)
                break

    return "\n".join(relevant_lines[:5])


def ask_rag(question):

    knowledge = load_knowledge()

    context = search_context(question, knowledge)

    prompt = f"""
Use ONLY the following knowledge.

Knowledge:
{context}

Question:
{question}

Answer:
"""

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]


while True:

    q = input("\nYou: ")

    if q.lower() == "exit":
        break

    answer = ask_rag(q)

    print("\nBot:", answer)