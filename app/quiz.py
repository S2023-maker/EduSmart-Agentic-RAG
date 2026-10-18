
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
    questions: List[QuizQuestion] = Field(min_length=5, max_length=5)


def extract_json(text: str) -> dict:
    """Extract a JSON object from the model response."""

    if not text or not text.strip():
        raise ValueError("The AI returned an empty response.")

    text = text.strip()

    # Remove optional Markdown code fences.
    text = re.sub(
        r"^\s*```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\s*```\s*$", "", text)

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    # Find a JSON object even if the model adds surrounding text.
    decoder = json.JSONDecoder()

    for match in re.finditer(r"\{", text):
        try:
            data, _ = decoder.raw_decode(text[match.start():])
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            continue

    raise ValueError(
        "The AI did not return valid quiz JSON. "
        f"Response preview: {text[:200]}"
    )


def validate_quiz(data: dict) -> Quiz:
    """Validate question count, options, and answer correctness."""

    quiz = Quiz.model_validate(data)

    for index, question in enumerate(quiz.questions, start=1):
        if not question.question.strip():
            raise ValueError(f"Question {index} is empty.")

        if len(question.options) != 4:
            raise ValueError(
                f"Question {index} must have exactly four options."
            )

        if any(
            not isinstance(option, str) or not option.strip()
            for option in question.options
        ):
            raise ValueError(
                f"Question {index} contains an empty or invalid option."
            )

        normalized_options = [
            option.strip().casefold()
            for option in question.options
        ]

        if len(set(normalized_options)) != 4:
            raise ValueError(
                f"Question {index} has duplicate options. "
                "All four options must be different."
            )

        normalized_answer = question.correct_answer.strip().casefold()

        if not any(
            normalized_answer == option.strip().casefold()
            for option in question.options
        ):
            raise ValueError(
                f"Question {index}: the correct answer must exactly "
                "match one of its four options."
            )

        if not question.explanation.strip():
            raise ValueError(
                f"Question {index} has no explanation."
            )

    return quiz


def get_response_text(response) -> str:
    """Handle common LangChain response content formats."""

    content = response.content

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []

        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)

        return "\n".join(parts)

    return str(content)


def generate_quiz_data(topic: str) -> Quiz:
    """Generate and validate a five-question multiple-choice quiz."""

    topic = topic.strip()

    if not topic:
        raise ValueError("Please enter a quiz topic.")

    documents = retrieve_documents(
        f"Grade 10 Science {topic}",
        k=5,
    )

    if not documents:
        raise ValueError(
            f"No curriculum content was found for: {topic}"
        )

    context = "\n\n".join(
        document.page_content
        for document in documents
        if document.page_content.strip()
    )

    if not context.strip():
        raise ValueError(
            f"No readable curriculum text was found for: {topic}"
        )

    context = context[:18000]

    prompt = f"""
You are an experienced Grade 10 Science teacher.

Create exactly FIVE multiple-choice questions about:
{topic}

Use the curriculum context below as your factual source.

CURRICULUM:
<context>
{context}
</context>

Return ONLY valid JSON using this structure:

{{
  "questions": [
    {{
      "question": "Write a clear question",
      "options": [
        "First distinct option",
        "Second distinct option",
        "Third distinct option",
        "Fourth distinct option"
      ],
      "correct_answer": "Exact text of one option",
      "explanation": "Brief factual explanation"
    }}
  ]
}}

STRICT REQUIREMENTS:
1. Return exactly 5 question objects.
2. Each question must have exactly 4 options.
3. All 4 options for each question must be genuinely different.
4. Never repeat an option using different capitalization or spacing.
5. Never use synonyms or equivalent answers as separate choices.
6. Make incorrect options plausible but factually incorrect.
7. Exactly one option must be correct for each question.
8. correct_answer must match the complete text of one option.
9. Every question and explanation must be non-empty.
10. Do not return Python objects, QuizQuestion(...), or questions as strings.
11. Do not repeat the same question.
12. Do not include Markdown fences or text outside the JSON.
13. Use double quotes for JSON strings and escape internal quotes.
14. Include no extra keys.

Check all four options for every question before returning the JSON.
"""

    llm = get_llm()
    previous_error = None

    # Retry generation up to three times.
    for attempt in range(1, 4):
        current_prompt = prompt

        if previous_error:
            current_prompt += f"""

IMPORTANT: Your previous attempt failed validation.

Validation error:
{previous_error}

Regenerate the ENTIRE quiz. Do not reuse the invalid options.
Check that every question has four unique, distinct options and that
the correct answer matches exactly one option.

Return only the corrected JSON object.
"""

        try:
            response = llm.invoke(current_prompt)
            content = get_response_text(response)

            data = extract_json(content)
            return validate_quiz(data)

        except Exception as exc:
            previous_error = str(exc)

    raise ValueError(
        "Quiz generation failed after three attempts. "
        f"Last validation error: {previous_error}"
    )
