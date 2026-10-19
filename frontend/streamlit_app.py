import sys
from pathlib import Path

import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
UPLOAD_DIR = PROJECT_ROOT / "uploaded_documents"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.agent import run_agent
from app.quiz import generate_quiz_data
from app.rag import answer_question
from app.document_loader import load_document, SUPPORTED_EXTENSIONS

st.set_page_config(
    page_title="EduSmart AI Assistant",
    page_icon="🎓",
    layout="wide",
)

DEMO_USERS = {
    "admin": {"password": "Admin123", "role": "Admin"},
    "student": {"password": "Student123", "role": "Student"},
    "employee": {"password": "Employee123", "role": "Employee"},
}

DEFAULTS = {
    "authenticated": False,
    "username": None,
    "user_role": None,
    "uploaded_vectorstore": None,
    "uploaded_documents": [],
    "processed_files": set(),
    "chat_history": [],
}

for key, default_value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default_value.copy() if isinstance(default_value, (list, set, dict)) else default_value

# ------------------------------
# Login
# ------------------------------
if not st.session_state.authenticated:
    st.title("🎓 EduSmart AI Assistant")
    st.subheader("Your AI-powered document and learning assistant")
    st.write("Upload documents, ask questions, generate summaries, and explore learning materials.")

    with st.form("login_form"):
        username = st.selectbox("Demo account", ["admin", "student", "employee"])
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login", use_container_width=True)

    if submitted:
        account = DEMO_USERS.get(username)
        if account and password == account["password"]:
            st.session_state.authenticated = True
            st.session_state.username = username
            st.session_state.user_role = account["role"]
            st.rerun()
        else:
            st.error("Incorrect password. Please try again.")

    with st.expander("Demo credentials"):
        st.markdown(
            "- **Admin:** `admin` / `Admin123`\n"
            "- **Student:** `student` / `Student123`\n"
            "- **Employee:** `employee` / `Employee123`"
        )
    st.stop()

username = st.session_state.username
role = st.session_state.user_role

st.sidebar.title("EduSmart")
st.sidebar.success(f"User: {username}")
st.sidebar.caption(f"Role: {role}")

if st.sidebar.button("Logout", use_container_width=True):
    st.session_state.authenticated = False
    st.session_state.username = None
    st.session_state.user_role = None
    st.rerun()

st.sidebar.divider()

# ------------------------------
# Document processing and retrieval
# ------------------------------
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def process_uploaded_files(uploaded_files):
    """Extract, split and index newly uploaded documents."""
    new_documents = []
    new_file_names = []

    for uploaded_file in uploaded_files:
        filename = Path(uploaded_file.name).name
        extension = Path(filename).suffix.lower()

        if extension not in SUPPORTED_EXTENSIONS:
            st.warning(f"Unsupported file type: {filename}")
            continue
        if filename in st.session_state.processed_files:
            continue

        destination = UPLOAD_DIR / filename
        try:
            destination.write_bytes(uploaded_file.getvalue())
            documents = load_document(str(destination))
            if not documents:
                st.warning(f"No readable text found in {filename}.")
                continue
            new_documents.extend(documents)
            new_file_names.append(filename)
        except Exception as exc:
            st.error(f"Could not process {filename}: {exc}")

    if not new_documents:
        if not new_file_names:
            st.info("No new readable documents were added.")
        return

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    new_chunks = splitter.split_documents(new_documents)
    if not new_chunks:
        st.warning("No text chunks were created from these documents.")
        return

    all_chunks = st.session_state.uploaded_documents + new_chunks
    try:
        with st.spinner("Creating searchable document index..."):
            vectorstore = FAISS.from_documents(all_chunks, get_embeddings())
        st.session_state.uploaded_documents = all_chunks
        st.session_state.uploaded_vectorstore = vectorstore
        st.session_state.processed_files.update(new_file_names)
        st.success(f"Processed {len(new_file_names)} file(s) successfully.")
    except Exception as exc:
        st.error(f"Could not create the document index: {exc}")


def get_uploaded_context(topic: str, k: int = 5) -> str:
    """Retrieve relevant text chunks from uploaded documents."""
    vectorstore = st.session_state.get("uploaded_vectorstore")
    if vectorstore is None:
        return ""

    try:
        documents = vectorstore.similarity_search(topic, k=k)
    except Exception as exc:
        st.warning(f"Could not retrieve uploaded document context: {exc}")
        return ""

    context_parts = []
    for index, document in enumerate(documents, start=1):
        source = Path(str(document.metadata.get("source", "Uploaded document"))).name
        page = document.metadata.get("page")
        location = f", page {page}" if page is not None else ""
        context_parts.append(
            f"[Source {index}: {source}{location}]\n{document.page_content}"
        )
    return "\n\n".join(context_parts)


def display_generated_result(result):
    """Display LangChain/Pydantic output safely in Streamlit."""
    if hasattr(result, "model_dump"):
        st.json(result.model_dump())
    elif isinstance(result, (dict, list)):
        st.json(result)
    else:
        st.markdown(str(result))

# ------------------------------
# Header and navigation
# ------------------------------
st.title("🎓 EduSmart AI Assistant")
st.caption("AI-powered learning, document search and knowledge assistance")
st.write(f"Welcome, **{username}**! You are logged in as **{role}**.")

if role == "Admin":
    modes = ["Ask EduSmart", "Upload Documents", "Take a Quiz", "Topic Summary", "Study Guide"]
elif role == "Student":
    modes = ["Ask EduSmart", "Take a Quiz", "Topic Summary", "Study Guide"]
else:
    modes = ["Ask EduSmart", "Topic Summary", "Study Guide"]

mode = st.sidebar.radio("Choose a feature", modes)

# ------------------------------
# Upload documents (Admin only)
# ------------------------------
if mode == "Upload Documents":
    st.header("📁 Document Library")
    st.write("Upload files to extract their text and create a searchable knowledge base.")
    st.info("Supported formats: PDF, DOCX, TXT, MD, CSV, XLSX and PPTX.")

    uploaded_files = st.file_uploader(
        "Choose documents",
        type=["pdf", "docx", "txt", "md", "csv", "xlsx", "pptx"],
        accept_multiple_files=True,
        key="multi_document_uploader",
    )

    if st.button("Process Documents", type="primary"):
        if uploaded_files:
            process_uploaded_files(uploaded_files)
        else:
            st.warning("Please select at least one document.")

    st.divider()
    st.subheader("Documents in this demo session")
    if st.session_state.processed_files:
        for filename in sorted(st.session_state.processed_files):
            st.write(f"📄 {filename}")
    else:
        st.info("No documents have been processed yet.")

# ------------------------------
# Ask EduSmart
# ------------------------------

elif mode == "Ask EduSmart":
    st.header("💬 Ask EduSmart")

    st.success(
        "Both knowledge sources are supported: "
        "your uploaded documents and the built-in curriculum."
    )

    if st.session_state.uploaded_vectorstore is not None:
        st.info(
            "Uploaded documents are indexed and ready. "
            "Answers can use relevant information from both sources."
        )
    else:
        st.info(
            "No uploaded documents are indexed in this session. "
            "Answers will use the built-in curriculum."
        )


    for item in st.session_state.chat_history:
        with st.chat_message(item["role"]):
            st.markdown(item["content"])

    question = st.text_area(
        "Enter your question",
        placeholder="Example: Summarize this document or explain its key points.",
        height=120,
        key="question_input",
    )

    if st.button("Get Answer", type="primary"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            try:
                with st.spinner("EduSmart is preparing your answer..."):
                    result = answer_question(
                        question.strip(),
                        vectorstore=st.session_state.uploaded_vectorstore,
                    )
                answer = str(result.get("answer", "No answer was returned."))
                st.session_state.chat_history.append({"role": "user", "content": question.strip()})
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                st.subheader("Answer")
                st.markdown(answer)

                sources = result.get("sources", [])
                if sources:
                    st.subheader("Sources")
                    for source in sources:
                        st.write(f"📄 {Path(str(source)).name}")

                documents = result.get("documents", [])
                if documents:
                    with st.expander("View retrieved document excerpts"):
                        for index, document in enumerate(documents, start=1):
                            metadata = document.metadata
                            source = metadata.get("source", "Document")
                            st.markdown(f"**Excerpt {index} — {Path(str(source)).name}**")
                            if "page" in metadata:
                                st.caption(f"Page: {metadata['page']}")
                            if "slide" in metadata:
                                st.caption(f"Slide: {metadata['slide']}")
                            if "sheet" in metadata:
                                st.caption(f"Worksheet: {metadata['sheet']}")
                            st.write(document.page_content)
            except Exception as exc:
                st.error(f"Unable to answer your question: {exc}")

# ------------------------------
# Take a Quiz — use uploaded context when available
# ------------------------------
elif mode == "Take a Quiz":
    st.header("📝 Take a Quiz")
    if st.session_state.uploaded_vectorstore is not None:
        st.caption("A quiz will be based on your uploaded documents when relevant content is retrieved.")

    topic = st.text_input("Enter a quiz topic", placeholder="Example: Python functions")
    if st.button("Generate Quiz", type="primary"):
        if not topic.strip():
            st.warning("Please enter a topic.")
        else:
            try:
                uploaded_context = get_uploaded_context(topic.strip())
                if uploaded_context:
                    quiz_topic = (
                        f"Topic: {topic.strip()}\n\n"
                        "Use the following uploaded document content as the primary source. "
                        "Create questions only from facts supported by this content. "
                        "Do not invent document-specific facts.\n\n"
                        f"SOURCE MATERIAL:\n{uploaded_context}"
                    )
                else:
                    quiz_topic = topic.strip()
                with st.spinner("Generating your quiz..."):
                    quiz_data = generate_quiz_data(quiz_topic)
                
                st.subheader("Generated Quiz")

                # Convert Quiz object to dictionary if necessary
                if hasattr(quiz_data, "model_dump"):
                    quiz_data = quiz_data.model_dump()
                elif hasattr(quiz_data, "dict") and not isinstance(quiz_data, dict):
                    quiz_data = quiz_data.dict()

                if isinstance(quiz_data, dict):
                    questions = quiz_data.get("questions", [])

                    if not questions:
                        st.warning("No questions were generated.")
                    else:
                        with st.form("interactive_quiz_form"):
                            answers = {}

                            for i, question in enumerate(questions):
                                st.markdown(f"### Question {i + 1}")
                                st.write(question.get("question", ""))

                                answers[i] = st.radio(
                                    "Select your answer:",
                                    question.get("options", []),
                                    index=None,
                                    key=f"quiz_answer_{i}",
                                )

                            submitted = st.form_submit_button(
                                "Submit Quiz",
                                type="primary",
                            )

                        if submitted:
                            score = 0

                            for i, question in enumerate(questions):
                                correct = question.get("correct_answer")
                                selected = answers.get(i)

                                if selected == correct:
                                    score += 1
                                    st.success(f"Question {i + 1}: Correct!")
                                else:
                                    st.error(
                                        f"Question {i + 1}: Incorrect. "
                                        f"Correct answer: {correct}"
                                    )

                                if question.get("explanation"):
                                    st.info(
                                        f"Explanation: {question['explanation']}"
                                    )

                            total = len(questions)
                            percentage = score / total * 100

                            st.markdown("## Quiz Results")
                            st.metric("Your Score", f"{score}/{total}")
                            st.progress(int(percentage))
                            st.write(f"Percentage: {percentage:.1f}%")
                else:
                    st.write(quiz_data)

            except Exception as exc:
                st.error(f"Could not generate quiz: {exc}")

# ------------------------------
# Topic Summary — use uploaded context when available
# ------------------------------
elif mode == "Topic Summary":
    st.header("📚 Topic Summary")
    topic = st.text_input("Enter a topic to summarize", placeholder="Example: Python functions")
    if st.button("Generate Summary", type="primary"):
        if not topic.strip():
            st.warning("Please enter a topic.")
        else:
            try:
                uploaded_context = get_uploaded_context(topic.strip())
                if uploaded_context:
                    prompt = f"""
Create a clear, structured summary of: {topic.strip()}.
Use the supplied uploaded document excerpts as your primary source.
Include key concepts, important points, and takeaways. Do not invent
facts or imply the document covers unsupported information. If the
excerpts do not cover the requested topic, say so clearly.

UPLOADED DOCUMENT EXCERPTS:
{uploaded_context}
"""
                else:
                    prompt = (
                        f"Create a clear, structured summary of: {topic.strip()}. "
                        "Include key concepts and important takeaways."
                    )
                with st.spinner("Preparing your summary..."):
                    result = run_agent(prompt)
                st.subheader("Summary")
                display_generated_result(result)
            except Exception as exc:
                st.error(f"Could not generate summary: {exc}")

# ------------------------------
# Study Guide — use uploaded context when available
# ------------------------------
elif mode == "Study Guide":
    st.header("📖 Study Guide")
    topic = st.text_input("Enter a topic for your study guide", placeholder="Example: Python functions")
    if st.button("Create Study Guide", type="primary"):
        if not topic.strip():
            st.warning("Please enter a topic.")
        else:
            try:
                uploaded_context = get_uploaded_context(topic.strip())
                if uploaded_context:
                    prompt = f"""
Create a beginner-friendly study guide for: {topic.strip()}.
Use the uploaded document excerpts below as the primary source.
Include learning objectives, key concepts, explanations, examples
supported by the material, revision notes, and five practice questions.
Do not invent document-specific facts. If the excerpts do not cover
the topic, state that clearly rather than pretending they do.

UPLOADED DOCUMENT EXCERPTS:
{uploaded_context}
"""
                else:
                    prompt = (
                        f"Create a beginner-friendly study guide for {topic.strip()}. "
                        "Include learning objectives, key concepts, examples, "
                        "revision notes, and five practice questions."
                    )
                with st.spinner("Creating your study guide..."):
                    result = run_agent(prompt)
                st.subheader("Your Study Guide")
                display_generated_result(result)
            except Exception as exc:
                st.error(f"Could not create study guide: {exc}")

st.divider()
st.caption("EduSmart AI Assistant | Demo application for college and office use")
