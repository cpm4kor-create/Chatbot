import chromadb
import ollama
from sentence_transformers import SentenceTransformer

MODEL_NAME = "qwen3"

# Load embedding model
embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# Load ChromaDB
client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_collection(
    "knowledge"
)


def retrieve_context(question):

    query_embedding = embedding_model.encode(
        question
    ).tolist()

    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    docs = result["documents"][0]

    return "\n".join(docs)


def ask_rag(question):

    context = retrieve_context(question)

    prompt = f"""
Answer only using the provided context.

Context:
{context}

Question:
{question}
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


def main():

    print("===== ChromaDB RAG Chatbot =====")
    print("Model:", MODEL_NAME)
    print("Type 'exit' to quit")

    while True:

        question = input("\nYou: ")

        if question.lower() == "exit":
            break

        answer = ask_rag(question)

        print("\nBot:", answer)


if __name__ == "__main__":
    main()