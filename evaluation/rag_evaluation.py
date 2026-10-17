import sys
import csv
import time
from pathlib import Path

# Add the project root to Python's import path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.rag import answer_question

REPORT_PATH = PROJECT_ROOT / "evaluation" / "rag_evaluation_report.csv"


TEST_QUESTIONS = [
    "What is the function of the human heart?",
    "Summarize the important points from the material.",
    "Explain one key concept in simple words.",
    "List three facts supported by the source documents.",
    "What is the serial number of the laptop used by the author?"
]


def evaluate_rag():
    results = []

    for index, question in enumerate(TEST_QUESTIONS, start=1):
        print(f"\nTest {index}/{len(TEST_QUESTIONS)}")
        print("Question:", question)

        start_time = time.perf_counter()

        try:
            response = answer_question(question)
            latency = round(time.perf_counter() - start_time, 2)

            # Your existing RAG function returns a dictionary.
            if isinstance(response, dict):
                answer = response.get("answer", str(response))
                sources = response.get("sources", [])
            else:
                answer = str(response)
                sources = []

            if isinstance(sources, (list, tuple)):
                sources_text = "; ".join(map(str, sources))
            else:
                sources_text = str(sources)

            status = "SUCCESS"
            error = ""

        except Exception as exc:
            latency = round(time.perf_counter() - start_time, 2)
            answer = ""
            sources_text = ""
            status = "FAILED"
            error = str(exc)

        print(f"Latency: {latency} seconds")
        print("Status:", status)
        print("Answer:", answer[:300])

        results.append({
            "question": question,
            "answer": answer,
            "sources": sources_text,
            "latency_seconds": latency,
            "status": status,
            "error": error,
            "relevance_score_1_to_5": "",
            "faithfulness_score_1_to_5": "",
            "review_notes": ""
        })

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(results[0].keys())

    with open(
        REPORT_PATH,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    successful = [r for r in results if r["status"] == "SUCCESS"]

    if successful:
        average_latency = sum(
            r["latency_seconds"] for r in successful
        ) / len(successful)

        print("\n--- Evaluation Summary ---")
        print(f"Total questions: {len(results)}")
        print(f"Successful: {len(successful)}")
        print(f"Failed: {len(results) - len(successful)}")
        print(f"Average latency: {average_latency:.2f} seconds")

    print(f"\nReport saved to:\n{REPORT_PATH}")


if __name__ == "__main__":
    evaluate_rag()
