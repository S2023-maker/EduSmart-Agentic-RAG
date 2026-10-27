
import json

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from app.llm import get_llm
from app.tools import EDUSMART_TOOLS


def call_model(state: MessagesState):
    """Let the LLM respond or request one or more tools."""
    llm = get_llm().bind_tools(EDUSMART_TOOLS)
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


def build_agent():
    """Build the EduSmart agent using LangGraph."""
    workflow = StateGraph(MessagesState)

    workflow.add_node("agent", call_model)
    workflow.add_node("tools", ToolNode(EDUSMART_TOOLS))

    workflow.add_edge(START, "agent")

    workflow.add_conditional_edges(
        "agent",
        tools_condition,
        {
            "tools": "tools",
            END: END,
        },
    )

    workflow.add_edge("tools", "agent")

    return workflow.compile()


def format_quiz_json(raw_content: str) -> str:
    """Convert quiz JSON into readable Markdown."""
    try:
        quiz = json.loads(raw_content)

        if not isinstance(quiz, dict) or "questions" not in quiz:
            return raw_content

        questions = quiz["questions"]
        if not isinstance(questions, list):
            return raw_content

        lines = ["## EduSmart Quiz", ""]

        for index, question in enumerate(questions, start=1):
            if not isinstance(question, dict):
                continue

            lines.append(
                f"### Question {index}: "
                f"{question.get('question', 'Question unavailable')}"
            )
            lines.append("")

            options = question.get("options", [])
            for option_index, option in enumerate(options):
                letter = chr(65 + option_index)
                lines.append(f"- **{letter}.** {option}")

            lines.append("")

            correct_answer = question.get("correct_answer")
            if correct_answer:
                lines.append(f"**Correct answer:** {correct_answer}")

            explanation = question.get("explanation")
            if explanation:
                lines.append(f"**Explanation:** {explanation}")

            lines.append("")

        return "\n".join(lines).strip()

    except (json.JSONDecodeError, TypeError, ValueError):
        # If this isn't quiz JSON, keep the original response unchanged.
        return raw_content



def run_agent(question: str, chat_history=None) -> str:
    """Run the EduSmart agent with short-term conversation memory."""
    if not question or not question.strip():
        return "Please enter a question."

    # Restore recent user and assistant messages.
    messages = []

    if chat_history:
        # Keep the most recent 10 messages to limit context size.
        for item in chat_history[-10:]:
            role = item.get("role")
            content = item.get("content", "")

            if not isinstance(content, str) or not content.strip():
                continue

            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))

    # Add the current question after the previous conversation.
    messages.append(HumanMessage(content=question.strip()))

    agent = build_agent()

    result = agent.invoke(
        {"messages": messages},
        config={"recursion_limit": 12},
    )

    # Print tool-selection diagnostics.
    for message in result["messages"]:
        for tool_call in getattr(message, "tool_calls", []):
            print(f"Tool selected: {tool_call['name']}")
            print(f"Tool arguments: {tool_call['args']}")

    # Preserve readable quiz formatting.
    for message in reversed(result["messages"]):
        if getattr(message, "name", None) == "generate_quiz":
            content = message.content

            if isinstance(content, str):
                formatted_quiz = format_quiz_json(content)
                if formatted_quiz != content:
                    return formatted_quiz

    final_message = result["messages"][-1]
    content = final_message.content

    if isinstance(content, str):
        return content

    return str(content)
