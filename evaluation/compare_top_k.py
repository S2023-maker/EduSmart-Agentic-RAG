
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

        for k in [2, 3, 4]:
            documents = retrieve_documents(
                query=question,
                k=k,
                vectorstore=vectorstore,
            )

            print(f"\n--- Top {k} chunks ---")

            for rank, doc in enumerate(documents, start=1):
                source = Path(
                    doc.metadata.get("source", "Unknown")
                ).name
                print(f"\nRank {rank} | {source}")
                print(doc.page_content[:350].replace("\n", " "))

if __name__ == "__main__":
    main()
