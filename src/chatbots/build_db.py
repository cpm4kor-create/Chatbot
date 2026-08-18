import chromadb
from sentence_transformers import SentenceTransformer

# Load embedding model
embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# Read knowledge file
with open(
    "knowledge.txt",
    "r",
    encoding="utf-8"
) as f:
    content = f.read()

# Split into chunks
chunks = [
    chunk.strip()
    for chunk in content.split("\n")
    if chunk.strip()
]

# Create ChromaDB client
client = chromadb.PersistentClient(
    path="./chroma_db"
)

# Create collection
collection = client.get_or_create_collection(
    name="knowledge"
)

# Store chunks
for i, chunk in enumerate(chunks):

    embedding = embedding_model.encode(
        chunk
    ).tolist()

    collection.add(
        ids=[str(i)],
        documents=[chunk],
        embeddings=[embedding]
    )

print("Knowledge base created successfully.")