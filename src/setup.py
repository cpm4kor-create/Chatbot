import subprocess
import sys

def install_requirements(file_name):
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-r", file_name]
    )

while True:

    print("\n========== AI PROJECT SETUP ==========")
    print("1. LLM Chatbot")
    print("2. RAG")
    print("3. Agentic AI")
    print("4. Exit")

    choice = input("Select option: ")

    if choice == "1":
        install_requirements("requirements/llm.txt")
        print("LLM dependencies installed.")
        break

    elif choice == "2":
        install_requirements("requirements/rag.txt")
        print("RAG dependencies installed.")
        break

    elif choice == "3":
        install_requirements("requirements/agentic.txt")
        print("Agentic AI dependencies installed.")
        break

    elif choice == "4":
        break

    else:
        print("Invalid option.")