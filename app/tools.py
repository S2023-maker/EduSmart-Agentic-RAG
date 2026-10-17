from langchain_core.tools import tool

from app.rag import answer_question


@tool
def retrieve_course_content(question: str) -> str:
    """
    Answer a student's question using the uploaded course curriculum.
    """

    result = answer_question(question)

    answer = result["answer"]

    if result["sources"]:
        answer += "\n\n**Sources:**\n"

        for source in result["sources"]:
            answer += f"- {source}\n"

    return answer


@tool
def general_tutor(question: str) -> str:
    """
    Answer general educational questions that do not require
    the uploaded curriculum.
    """

    from app.llm import get_llm

    llm = get_llm()

    response = llm.invoke(
        f"""
You are EduSmart, a helpful educational tutor.

Answer the student's question in simple and clear language.
Use examples when useful.

Student question:
{question}
"""
    )

    return str(response.content)


@tool
def generate_quiz(topic: str) -> str:
    """
    Generate a 5-question quiz using the uploaded curriculum.

    This tool returns the quiz as JSON so the Streamlit frontend
    can display interactive MCQs.
    """

    from app.quiz import generate_quiz_data

    quiz = generate_quiz_data(topic)

    return quiz.model_dump_json()


@tool
def summarize_topic(topic: str) -> str:
    """
    Create a simple summary of a topic using the course curriculum.
    """

    result = answer_question(
        f"Give the important curriculum content about {topic}"
    )

    from app.llm import get_llm

    llm = get_llm()

    response = llm.invoke(
        f"""
Create a simple Grade 10 study summary.

Topic:
{topic}

Curriculum content:
{result["answer"]}

Include:
- Key concepts
- Important definitions
- Important facts
- Simple examples where useful

Use only the provided curriculum content.
"""
    )

    return str(response.content)


@tool
def create_study_guide(topic: str) -> str:
    """
    Create a study guide for a curriculum topic.
    """

    result = answer_question(
        f"Identify the important topics and concepts related to {topic}"
    )

    from app.llm import get_llm

    llm = get_llm()

    response = llm.invoke(
        f"""
You are an educational study-planning assistant.

Create a practical study guide for:

Topic:
{topic}

Relevant curriculum information:
{result["answer"]}

Include:

1. What to learn first
2. Important concepts
3. What to revise
4. Practice questions to focus on
5. A simple revision checklist

Keep it suitable for a Grade 10 student.
Use only information supported by the curriculum.
"""
    )

    return str(response.content)


EDUSMART_TOOLS = [
    retrieve_course_content,
    generate_quiz,
    summarize_topic,
    create_study_guide,
    general_tutor,
]
