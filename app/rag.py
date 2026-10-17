
from langchain_core.documents import Document

from app.llm import get_llm
from app.prompts import RAG_PROMPT
from app.retriever import retrieve_documents


def format_documents(documents: list[Document]) -> str:
    """
    Format retrieved documents into context for the LLM.
    Includes source and page metadata where available.
    """
    if not documents:
        return "No relevant content was found in the provided documents."

    formatted_chunks = []

    for i, doc in enumerate(documents, start=1):
        source = doc.metadata.get("source", "Unknown source")
        page = doc.metadata.get("page", "Unknown")

        # Display only the filename when a full local path is available.
        if isinstance(source, str) and "\\" in source:
            source = source.split("\\")[-1]
        elif isinstance(source, str) and "/" in source:
            source = source.split("/")[-1]

        chunk = (
            f"[Source {i}]\n"
            f"Document: {source}\n"
            f"Page: {page}\n"
            f"Content:\n{doc.page_content}"
        )

        formatted_chunks.append(chunk)

    return "\n\n".join(formatted_chunks)


def answer_question(
    question: str,
    k: int = 3,
    vectorstore=None,
) -> dict:
    """
    Retrieve relevant documents and generate a grounded answer.

    Returns:
        {
            "answer": str,
            "sources": list[str],
            "documents": list[Document]
        }
    """
    if not isinstance(question, str) or not question.strip():
        return {
            "answer": "Please enter a question.",
            "sources": [],
            "documents": [],
        }

    # Retrieve the most relevant document chunks.
    documents = retrieve_documents(
        question,
        k=k,
        vectorstore=vectorstore,
    )

    # Format retrieved content for the language model.
    context = format_documents(documents)

    # Strong grounding instructions to reduce incorrect refusals.
    prompt = f"""
You are EduSmart AI Tutor, a helpful and accurate science tutor.

Answer the student's question using the supplied document context.

IMPORTANT RULES:
1. Read every supplied context passage before answering.
2. Use relevant facts explicitly stated anywhere in the context.
3. Do not say information is missing when a passage contains it.
4. Answer directly in simple, student-friendly language.
5. Do not invent facts or present outside knowledge as if it came
   from the supplied documents.
6. If the documents support only part of an answer, explain that part
   and clearly identify what the documents do not explain.
7. If the answer is genuinely absent from all supplied passages,
   politely say that it was not found in the provided documents.
8. Distinguish between what the textbook states and any mechanism
   or explanation that the textbook does not provide.
9. Mention the relevant chapter or source when possible.

DOCUMENT CONTEXT:
{context}

STUDENT'S QUESTION:
{question}

Write a clear, accurate answer based on the context above.
"""

    try:
        response = get_llm().invoke(prompt)
        answer = (
            response.content.strip()
            if isinstance(response.content, str)
            else str(response.content).strip()
        )

        if not answer:
            answer = (
                "I could not generate an answer. "
                "Please try asking the question again."
            )

    except Exception as exc:
        print(f"Error generating answer: {exc}")
        answer = (
            "Sorry, I encountered an error while generating the answer. "
            "Please try again."
        )

    # Collect unique source documents.
    sources = []
    seen_sources = set()

    for doc in documents:
        source = doc.metadata.get("source", "Unknown source")

        if source not in seen_sources:
            seen_sources.add(source)
            sources.append(source)

    return {
        "answer": answer,
        "sources": sources,
        "documents": documents,
    }


def ask_tutor(question: str, vectorstore=None) -> dict:
    """
    Backward-compatible helper for the tutor interface.
    """
    return answer_question(
        question=question,
        vectorstore=vectorstore,
    )
