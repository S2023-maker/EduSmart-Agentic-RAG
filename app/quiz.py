import json
import re
from typing import List

from pydantic import BaseModel, Field

from app.llm import get_llm
from app.retriever import retrieve_documents


class QuizQuestion(BaseModel):
    question: str
    options: List[str] = Field(min_length=4, max_length=4)
    correct_answer: str
    explanation: str


class Quiz(BaseModel):
    questions: List[QuizQuestion] = Field(
        min_length=1,
        max_length=5
    )


def generate_quiz_data(topic: str) -> Quiz:

    # Retrieve curriculum content directly.
    # This avoids the old two-LLM-call quiz process.
    documents = retrieve_documents(
        f"Grade 10 Science {topic}",
        k=5,
    )

    if not documents:
        raise ValueError(
            f"No curriculum content found for topic: {topic}"
        )

    context_parts = []

    for doc in documents:
        context_parts.append(doc.page_content)

    context = "\n\n".join(context_parts)

    llm = get_llm()

    prompt = f"""
You are EduSmart, a Grade 10 Science quiz generator.

Create exactly 5 multiple-choice questions about:

TOPIC:
{topic}

Use ONLY the curriculum context below.

CURRICULUM:
{context}

Rules:

1. Create exactly 5 questions.
2. Each question must have exactly 4 options.
3. Each option must be a string.
4. correct_answer must exactly match one of the options.
5. Give a short explanation for every answer.
6. Questions must be suitable for Grade 10 students.
7. Do not invent information.
8. Do not use Markdown.
9. Return ONLY valid JSON.

Required format:

{{
    "questions": [
        {{
            "question": "Question text",
            "options": [
                "Option 1",
                "Option 2",
                "Option 3",
                "Option 4"
            ],
            "correct_answer": "Option 1",
            "explanation": "Short explanation"
        }}
    ]
}}
"""

    response = llm.invoke(prompt)

    content = str(response.content).strip()

    # Remove accidental markdown code fences
    content = re.sub(
        r"^```json\s*",
        "",
        content,
        flags=re.IGNORECASE,
    )

    content = re.sub(
        r"^```\s*",
        "",
        content,
    )

    content = re.sub(
        r"\s*```$",
        "",
        content,
    )

    # Extract JSON if the model added extra text
    start = content.find("{")
    end = content.rfind("}")

    if start == -1 or end == -1:
        raise ValueError(
            "The AI did not return valid quiz JSON."
        )

    content = content[start:end + 1]

    try:
        data = json.loads(content)

    except json.JSONDecodeError as e:

        raise ValueError(
            f"Could not parse quiz JSON: {e}"
        )

    quiz = Quiz.model_validate(data)

    # Final validation
    if len(quiz.questions) != 5:
        raise ValueError(
            f"Expected 5 questions but received "
            f"{len(quiz.questions)}."
        )

    for question in quiz.questions:

        if len(question.options) != 4:
            raise ValueError(
                "Every question must have exactly 4 options."
            )

        if question.correct_answer not in question.options:
            raise ValueError(
                "Correct answer must match one of the options."
            )

    return quiz

