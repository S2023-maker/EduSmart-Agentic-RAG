
import time
from app.rag import answer_question

questions = [
    "What is the main topic of the uploaded document?",
    "List three facts explicitly mentioned in the document.",
    "If the answer is not in the document, say you cannot find it."
]

for question in questions:
    start = time.perf_counter()

    try:
        answer = answer_question(question)
        elapsed = time.perf_counter() - start

        print("\nQuestion:", question)
        print(f"Response time: {elapsed:.2f} seconds")
        print("Answer:", str(answer)[:500])

    except Exception as e:
        elapsed = time.perf_counter() - start
        print("\nQuestion:", question)
        print(f"Failed after: {elapsed:.2f} seconds")
        print("Error:", str(e))
