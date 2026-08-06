import ast
import datetime
import operator
import os

import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer


load_dotenv()

CHROMA_PATH = os.getenv("CHROMA_PATH", "chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "agentic_kb")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)


def search_knowledge_base(query: str, top_k: int = 3) -> str:
    """
    Search the ChromaDB knowledge base using semantic similarity.
    """

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    try:
        count = collection.count()
    except Exception:
        count = 0

    if count == 0:
        return "Knowledge base is empty. Please run python build_db.py first."

    query_embedding = embedding_model.encode([query]).tolist()[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    documents = results.get("documents", [[]])[0]

    if not documents:
        return "No relevant information found in the knowledge base."

    context = "\n\n".join(
        [
            f"Context {index + 1}:\n{doc}"
            for index, doc in enumerate(documents)
        ]
    )

    return context


def get_current_date() -> str:
    """
    Return current local date and time.
    """

    now = datetime.datetime.now()

    return now.strftime(
        "Current date and time: %A, %d %B %Y, %I:%M %p"
    )


_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Only numbers are allowed.")

    if isinstance(node, ast.BinOp):
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        op_type = type(node.op)

        if op_type not in _ALLOWED_OPERATORS:
            raise ValueError("Operator not allowed.")

        return _ALLOWED_OPERATORSleft, right

    if isinstance(node, ast.UnaryOp):
        operand = _safe_eval(node.operand)
        op_type = type(node.op)

        if op_type not in _ALLOWED_OPERATORS:
            raise ValueError("Unary operator not allowed.")

        return _ALLOWED_OPERATORSoperand

    raise ValueError("Invalid math expression.")


def calculate_math(expression: str) -> str:
    """
    Safely calculate a math expression.
    Example: 25 * 18 + 5
    """

    cleaned = (
        expression
        .replace("calculate", "")
        .replace("Calculate", "")
        .replace("what is", "")
        .replace("What is", "")
        .replace("?", "")
        .strip()
    )

    try:
        tree = ast.parse(cleaned, mode="eval")
        result = _safe_eval(tree.body)
        return f"Calculation result: {result}"
    except Exception as e:
        return f"Could not calculate the expression. Error: {e}"
