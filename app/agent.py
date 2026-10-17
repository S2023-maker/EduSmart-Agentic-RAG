from app.llm import get_llm
from app.tools import (
    retrieve_course_content,
    generate_quiz,
    summarize_topic,
    create_study_guide,
    general_tutor,
)


def run_agent(question: str):
    """
    EduSmart Agentic Router.

    The agent decides which tool should handle the student's request.
    """

    llm = get_llm()

    router_prompt = f"""
You are the routing agent for EduSmart.

Choose exactly ONE action for the student's request.

Available actions:

CURRICULUM
- Questions about Grade 10 Science
- Questions about concepts in the uploaded curriculum
- Chapter-related questions

QUIZ
- Requests for quizzes
- MCQs
- Tests
- Practice questions
- "Quiz me"

SUMMARY
- Requests to summarize a topic
- Short notes
- Revision notes
- Key points

STUDY_GUIDE
- Requests for study plans
- What to study
- How to prepare
- Revision plans

GENERAL
- General educational questions
- Questions unrelated to the uploaded curriculum
- General learning advice

Student question:
{question}

Reply with EXACTLY ONE action:

CURRICULUM
QUIZ
SUMMARY
STUDY_GUIDE
GENERAL
"""

    response = llm.invoke(router_prompt)

    decision = str(response.content).strip().upper()

    print(f"Agent decision: {decision}")

    # -----------------------------
    # Curriculum RAG
    # -----------------------------
    if "CURRICULUM" in decision:
        return retrieve_course_content.invoke(
            {"question": question}
        )

    # -----------------------------
    # Quiz Generator
    # -----------------------------
    elif "QUIZ" in decision:
        return generate_quiz.invoke(
            {"topic": question}
        )

    # -----------------------------
    # Topic Summary
    # -----------------------------
    elif "SUMMARY" in decision:
        return summarize_topic.invoke(
            {"topic": question}
        )

    # -----------------------------
    # Study Guide
    # -----------------------------
    elif "STUDY_GUIDE" in decision:
        return create_study_guide.invoke(
            {"topic": question}
        )

    # -----------------------------
    # General Tutor
    # -----------------------------
    else:
        return general_tutor.invoke(
            {"question": question}
        )


if __name__ == "__main__":

    question = input("Student: ")

    print("\nEduSmart:")
    print(run_agent(question))