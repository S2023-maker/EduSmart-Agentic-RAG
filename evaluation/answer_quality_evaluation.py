
import csv
import json
import time
import sys
from pathlib import Path

# Make the project root importable.
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from app.rag import answer_question

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "eval_dataset.json"
REPORT_PATH = BASE_DIR / "answer_quality_report.csv"


def evaluate_question(item):
    question = item["question"]
    expected_behavior = item["expected_behavior"]
    expected_topic = item["expected_topic"]

    start_time = time.perf_counter()

    try:
        result = answer_question(question)

        latency = round(time.perf_counter() - start_time, 2)
        answer = str(result.get("answer") or "").strip()
        sources = result.get("sources") or []

        # Never treat an empty answer as a successful response.
        if not answer:
            return {
                "question": question,
                "expected_behavior": expected_behavior,
                "answer": "",
                "sources": "; ".join(sources),
                "latency_seconds": latency,
                "status": "EMPTY_ANSWER",
                "behavior_result": "FAIL",
                "declined": False,
                "topic_keyword_coverage": 0.0,
                "topic_keywords_matched": "",
                "review_notes": "The RAG pipeline returned an empty answer.",
                "error": "Empty answer returned.",
            }

        answer_lower = answer.lower()

        # Basic topic keyword coverage (not a semantic accuracy score).
        topic_keywords = [
            word.lower()
            for word in expected_topic.split()
            if len(word) > 3
        ]

        topic_matches = [
            word for word in topic_keywords
            if word in answer_lower
        ]

        topic_coverage = (
            round(len(topic_matches) / len(topic_keywords), 2)
            if topic_keywords else 0.0
        )

        
       
        
        refusal_phrases = [
            "does not contain",
            "does not include",
            "not mentioned in",
            "not available in",
            "not found in",
            "cannot answer",
            "can't answer",
            "unable to answer",
            "not covered in",
            "not covered by",
            "don't have enough information",
            "do not have enough information",
            "insufficient information",
            "not provided in the context",
            "none of the supplied documents mention",
            "documents you provided do not contain",
            "not present in the provided documents",
            "not present in the supplied documents",
        ]

        declined = any(
            phrase in answer_lower
            for phrase in refusal_phrases
        )

        

        

        declined = any(
            phrase in answer_lower
            for phrase in refusal_phrases
        )

        if expected_behavior == "answer":
            behavior_result = (
                "PASS" if not declined else "REVIEW"
            )
        else:
            behavior_result = (
                "PASS" if declined else "REVIEW"
            )

        return {
            "question": question,
            "expected_behavior": expected_behavior,
            "answer": answer,
            "sources": "; ".join(sources),
            "latency_seconds": latency,
            "status": "SUCCESS",
            "behavior_result": behavior_result,
            "declined": declined,
            "topic_keyword_coverage": topic_coverage,
            "topic_keywords_matched": "; ".join(topic_matches),
            "review_notes": (
                "Manually verify factual accuracy and source support."
            ),
            "error": "",
        }

    except Exception as exc:
        return {
            "question": question,
            "expected_behavior": expected_behavior,
            "answer": "",
            "sources": "",
            "latency_seconds": round(
                time.perf_counter() - start_time, 2
            ),
            "status": "ERROR",
            "behavior_result": "FAIL",
            "declined": False,
            "topic_keyword_coverage": 0.0,
            "topic_keywords_matched": "",
            "review_notes": "Evaluation raised an exception.",
            "error": f"{type(exc).__name__}: {exc}",
        }


def main():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    with DATASET_PATH.open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    if not dataset:
        raise ValueError("Evaluation dataset is empty.")

    results = []

    print(f"Evaluating {len(dataset)} questions...\n")

    for index, item in enumerate(dataset, start=1):
        print(f"[{index}/{len(dataset)}] {item['question']}")

        result = evaluate_question(item)
        results.append(result)

        print(
            f"Status: {result['status']} | "
            f"Behavior: {result['behavior_result']} | "
            f"Latency: {result['latency_seconds']}s\n"
        )

    with REPORT_PATH.open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(results[0].keys()),
        )
        writer.writeheader()
        writer.writerows(results)

    successful = sum(
        row["status"] == "SUCCESS" for row in results
    )
    behavior_passes = sum(
        row["behavior_result"] == "PASS" for row in results
    )
    empty_answers = sum(
        row["status"] == "EMPTY_ANSWER" for row in results
    )

    print("=" * 50)
    print("ANSWER QUALITY EVALUATION")
    print("=" * 50)
    print(f"Total questions: {len(results)}")
    print(f"Successful responses: {successful}/{len(results)}")
    print(f"Empty answers: {empty_answers}")
    print(f"Expected-behavior checks passed: {behavior_passes}/{len(results)}")
    print(f"CSV report: {REPORT_PATH}")
    print("\nNote: Manually verify answer relevance and faithfulness.")


if __name__ == "__main__":
    main()
