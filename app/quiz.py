import json
import re
from typing import List

from pydantic import BaseModel, Field, ValidationError

from app.llm import get_llm
from app.retriever import retrieve_documents


# ============================================================
# QUIZ DATA MODELS
# ============================================================

class QuizQuestion(BaseModel):
    question: str = Field(min_length=1)
    options: List[str] = Field(min_length=4, max_length=4)
    correct_answer: str = Field(min_length=1)
    explanation: str = Field(min_length=1)


class Quiz(BaseModel):
    questions: List[QuizQuestion] = Field(min_length=5, max_length=5)


# ============================================================
# JSON EXTRACTION
# ============================================================

def extract_json(text: str) -> dict:
    """Extract a JSON object from plain JSON or a fenced LLM response."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("The model returned an empty response.")

    cleaned = text.strip()
    cleaned = re.sub(r"^\s*```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```\s*$", "", cleaned)

    decoder = json.JSONDecoder()
    last_error = None

    # Try each opening brace, in case the model added introductory text.
    for match in re.finditer(r"\{", cleaned):
        try:
            value, _end = decoder.raw_decode(cleaned[match.start():])
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError as exc:
            last_error = exc

    raise ValueError(f"Could not find a valid JSON object in the model response. {last_error or ''}")

# ============================================================
# VALIDATION HELPERS
# ============================================================

def _validate_quiz_data(data: dict) -> Quiz:
    """Validate structure and ensure the answer is one of the options."""
    quiz = Quiz.model_validate(data)

    for index, question in enumerate(quiz.questions, start=1):
        options = [option.strip() for option in question.options]
        if any(not option for option in options):
            raise ValueError(f"Question {index} contains an empty option.")
        if len({option.casefold() for option in options}) != 4:
            raise ValueError(f"Question {index} must have four distinct options.")

        if not any(
            question.correct_answer.strip().casefold() == option.casefold()
            for option in options
        ):
            raise ValueError(
                f"Question {index}: correct_answer must exactly match one of its four options."
            )

    return quiz

# ============================================================
# QUIZ GENERATION
# ============================================================

def generate_quiz_data(topic: str) -> Quiz:
    """
    Generate exactly five multiple-choice questions using retrieved context.
    The model is retried when its response is invalid.
    """
    if not isinstance(topic, str) or not topic.strip():
        raise ValueError("Please enter a quiz topic.")

    topic = topic.strip()
    documents = retrieve_documents(f"Grade 10 Science {topic}", k=5)

    if not documents:
        raise ValueError(
            "No relevant content was found for this topic. Try a more specific topic "
            "or check that the curriculum FAISS index has been created."
        )

    context = "\n\n".join(
        str(getattr(document, "page_content", "")).strip()
        for document in documents
        if getattr(document, "page_content", "").strip()
    )[:18000]

    if not context.strip():
        raise ValueError("Retrieved documents did not contain readable text.")

    prompt = f"""
You are an educational quiz writer. Use ONLY the source context below.
The requested topic is: {topic}

Create exactly 5 multiple-choice questions suitable for a learner.
Requirements:
- Every question must be supported by the source context.
- Each question must have exactly 4 distinct answer options.
- Each correct_answer must match one option exactly, character for character.
- Provide one correct answer and a concise explanation for every question.
- Do not use the same question twice.
- Return ONLY a valid JSON object, with no Markdown fences or extra commentary.
- Use this exact schema:
{{
  "questions": [
    {{
      "question": "Question text",
      "options": ["Option A", "Option B", "Option C", "Option D"],
      "correct_answer": "Exact text of one option",
      "explanation": "Why that answer is correct according to the source"
    }}
  ]
}}

SOURCE CONTEXT:
<context>
{context}
</context>
"""

    errors = []
    for attempt in range(1, 4):
        try:
            response = get_llm().invoke(prompt)
            raw_text = (
                response.content
                if hasattr(response, "content")
                else str(response)
            )

            if isinstance(raw_text, list):
                raw_text = "\n".join(
                    str(part.get("text", part)) if isinstance(part, dict) else str(part)
                    for part in raw_text
                )

            data = extract_json(str(raw_text))
            return _validate_quiz_data(data)

        except (ValueError, ValidationError, json.JSONDecodeError) as exc:
            errors.append(f"Attempt {attempt}: {exc}")
        except Exception as exc:
            errors.append(f"Attempt {attempt}: {exc}")

    raise RuntimeError(
        "Could not generate a valid quiz after 3 attempts. "
        + " | ".join(errors)
    )
   