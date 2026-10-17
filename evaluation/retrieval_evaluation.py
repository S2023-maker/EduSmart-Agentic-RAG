
import sys
import csv
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.retriever import get_vectorstore, retrieve_documents

K = 4
REPORT_PATH = PROJECT_ROOT / "evaluation" / "retrieval_evaluation_report.csv"

# Each question has a known relevant curriculum chapter.
TEST_CASES = [
    {
        "question": "What is the function of the human heart?",
        "expected_source": "chapter5",
    },
    {
        "question": "How do you write a balanced chemical equation?",
        "expected_source": "chapter1",
    },
    {
        "question": "How do CFCs affect the ozone layer?",
        "expected_source": "chapter13",
    },
    {
        "question": "Explain the reactivity series of metals.",
        "expected_source": "chapter3",
    },
    {
        "question": "How does carbon burn in oxygen?",
        "expected_source": "chapter4",
    },
]


def normalize_source(source):
    """Normalize a source path or filename for chapter matching."""
    return re.sub(r"[^a-z0-9]", "", str(source).lower())


def is_relevant(source, expected_source):
    """Check whether a retrieved chunk belongs to the expected chapter."""
    return expected_source in normalize_source(source)


def evaluate_retrieval():
    vectorstore = get_vectorstore()
    results = []

    for number, test in enumerate(TEST_CASES, start=1):
        question = test["question"]
        expected_source = test["expected_source"]

        print(f"\nTest {number}/{len(TEST_CASES)}")
        print("Question:", question)
        print("Expected chapter:", expected_source)

        documents = retrieve_documents(
            query=question,
            k=K,
            vectorstore=vectorstore,
        )

        relevant_flags = []
        retrieved_sources = []

        for rank, doc in enumerate(documents, start=1):
            source = doc.metadata.get("source", "Unknown")
            relevant = is_relevant(source, expected_source)

            relevant_flags.append(relevant)
            retrieved_sources.append(str(source))

            print(
                f"  Rank {rank}: {source} | "
                f"Relevant: {'Yes' if relevant else 'No'}"
            )

        # Precision@K: proportion of retrieved chunks from expected chapter.
        precision_at_k = sum(relevant_flags) / K

        # Chapter-level recall: was the expected chapter retrieved?
        # Each test case specifies one expected relevant chapter.
        recall_at_k = (
            1.0 if any(relevant_flags) else 0.0
        )

        # Reciprocal rank of the first chunk from the expected chapter.
        reciprocal_rank = 0.0
        for rank, relevant in enumerate(relevant_flags, start=1):
            if relevant:
                reciprocal_rank = 1.0 / rank
                break

        results.append({
            "question": question,
            "expected_chapter": expected_source,
            "retrieved_sources": " | ".join(retrieved_sources),
            "retrieved_chunks": len(documents),
            "precision_at_4": round(precision_at_k, 3),
            "chapter_recall_at_4": round(recall_at_k, 3),
            "reciprocal_rank": round(reciprocal_rank, 3),
        })

        print(f"Precision@{K}: {precision_at_k:.3f}")
        print(f"Chapter Recall@{K}: {recall_at_k:.3f}")
        print(f"Reciprocal Rank: {reciprocal_rank:.3f}")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(
        REPORT_PATH,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(results[0].keys()),
        )
        writer.writeheader()
        writer.writerows(results)

    print("\n========== EVALUATION SUMMARY ==========")

    count = len(results)

    print(
        "Average Precision@4:",
        round(sum(r["precision_at_4"] for r in results) / count, 3),
    )
    print(
        "Average Chapter Recall@4:",
        round(sum(r["chapter_recall_at_4"] for r in results) / count, 3),
    )
    print(
        "Mean Reciprocal Rank (MRR):",
        round(sum(r["reciprocal_rank"] for r in results) / count, 3),
    )

    print("\nReport saved to:", REPORT_PATH)


if __name__ == "__main__":
    evaluate_retrieval()
