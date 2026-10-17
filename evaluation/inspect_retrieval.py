
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.retriever import get_vectorstore, retrieve_documents

QUESTIONS = [
    "What is the function of the human heart?",
    "How does carbon burn in oxygen?",
]

def main():
    vectorstore = get_vectorstore()

    for question in QUESTIONS:
        print("\n" + "=" * 80)
        print("QUESTION:", question)
        print("=" * 80)

        documents = retrieve_documents(
            query=question,
            k=4,
            vectorstore=vectorstore,
        )

        for rank, doc in enumerate(documents, start=1):
            print(f"\n--- Retrieved chunk {rank} ---")
            print("Source:", doc.metadata.get("source", "Unknown"))
            print("Page:", doc.metadata.get("page", "Unknown"))
            print("Text:")
            print(doc.page_content[:1200])

if __name__ == "__main__":
    main()
