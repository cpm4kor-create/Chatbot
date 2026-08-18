import os
import re
from typing import TypedDict

import ollama
from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph

from tools import calculate_math, get_current_date, search_knowledge_base


load_dotenv()

MODEL_NAME = os.getenv("OLLAMA_MODEL", "qwen3:latest")


class AgentState(TypedDict, total=False):
    question: str
    route: str
    tool_result: str
    final_answer: str


def router_node(state: AgentState) -> AgentState:
    """
    Decide which tool should be used.
    This is the agent decision node.
    """

    question = state["question"].lower()

    date_keywords = [
        "date",
        "today",
        "day",
        "time",
        "month",
        "year",
        "current date",
        "current time"
    ]

    math_keywords = [
        "calculate",
        "math",
        "solve",
        "plus",
        "minus",
        "multiply",
        "divide",
        "percentage",
        "sum"
    ]

    math_pattern = r"[\d\s\+\-\*\/\%\(\)\.]+"

    if any(keyword in question for keyword in date_keywords):
        route = "date"

    elif any(keyword in question for keyword in math_keywords):
        route = "calculator"

    elif re.fullmatch(math_pattern, question.strip()):
        route = "calculator"

    elif any(operator in question for operator in ["+", "-", "*", "/", "%"]):
        route = "calculator"

    else:
        route = "knowledge_base"

    return {
        "question": state["question"],
        "route": route
    }


def date_node(state: AgentState) -> AgentState:
    result = get_current_date()

    return {
        **state,
        "tool_result": result
    }


def calculator_node(state: AgentState) -> AgentState:
    result = calculate_math(state["question"])

    return {
        **state,
        "tool_result": result
    }


def knowledge_base_node(state: AgentState) -> AgentState:
    result = search_knowledge_base(state["question"])

    return {
        **state,
        "tool_result": result
    }


def final_answer_node(state: AgentState) -> AgentState:
    """
    Generate the final answer using Ollama.
    """

    route = state["route"]
    question = state["question"]
    tool_result = state.get("tool_result", "")

    if route == "date":
        prompt = f"""
User question:
{question}

Tool result:
{tool_result}

Answer the user clearly and shortly.
"""

    elif route == "calculator":
        prompt = f"""
User question:
{question}

Calculator result:
{tool_result}

Give the final answer clearly.
"""

    elif route == "knowledge_base":
        prompt = f"""
You are a helpful agentic AI chatbot.

Answer the user using the knowledge base context below.
If the context does not contain enough information, say that the knowledge base does not contain enough information.

User question:
{question}

Knowledge base context:
{tool_result}

Final answer:
"""

    else:
        prompt = question

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    answer = response["message"]["content"]

    return {
        **state,
        "final_answer": answer
    }


def route_condition(state: AgentState) -> str:
    return state["route"]


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("router", router_node)
    graph.add_node("date", date_node)
    graph.add_node("calculator", calculator_node)
    graph.add_node("knowledge_base", knowledge_base_node)
    graph.add_node("final_answer", final_answer_node)

    graph.add_edge(START, "router")

    graph.add_conditional_edges(
        "router",
        route_condition,
        {
            "date": "date",
            "calculator": "calculator",
            "knowledge_base": "knowledge_base"
        }
    )

    graph.add_edge("date", "final_answer")
    graph.add_edge("calculator", "final_answer")
    graph.add_edge("knowledge_base", "final_answer")
    graph.add_edge("final_answer", END)

    return graph.compile()


def terminal_chatbot():
    app = build_graph()

    print("===== Agentic AI Terminal Chatbot =====")
    print("Model:", MODEL_NAME)
    print("This chatbot can:")
    print("1. Search knowledge base")
    print("2. Calculate math")
    print("3. Check current date/time")
    print("4. Give final answer")
    print("Type 'exit' to quit.")

    while True:
        question = input("\nYou: ").strip()

        if question.lower() == "exit":
            print("Goodbye!")
            break

        if not question:
            continue

        try:
            result = app.invoke(
                {
                    "question": question
                }
            )

            print("\nRoute selected:", result["route"])
            print("\nBot:", result["final_answer"])

        except Exception as e:
            print("\nError:")
            print(e)
            print("\nPlease check:")
            print("1. Ollama is running")
            print("2. Model exists in ollama list")
            print("3. ChromaDB was built using python build_db.py")
            print("4. Required packages are installed")


if __name__ == "__main__":
    terminal_chatbot()