import sys
from pathlib import Path

import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ------------------------------------------------------------
# PROJECT PATHS
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
UPLOAD_DIR = PROJECT_ROOT / "uploaded_documents"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ------------------------------------------------------------
# PROJECT IMPORTS
# ------------------------------------------------------------

from app.agent import run_agent
from app.quiz import generate_quiz_data
from app.rag import answer_question
from app.document_loader import load_document, SUPPORTED_EXTENSIONS
from app.chat_history import (
    init_db,
    create_conversation,
    save_message,
    get_conversations,
    get_messages,
)

# Initialize SQLite database
init_db()

# ------------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------------

st.set_page_config(
    page_title="EduSmart AI Assistant",
    page_icon="🎓",
    layout="wide",
)

# ------------------------------------------------------------
# DEMO ACCOUNTS
# ------------------------------------------------------------

DEMO_USERS = {
    "admin": {"password": "Admin123", "role": "Admin"},
    "student": {"password": "Student123", "role": "Student"},
    "employee": {"password": "Employee123", "role": "Employee"},
}

# ------------------------------------------------------------
# SESSION STATE
# ------------------------------------------------------------

DEFAULTS = {
    "authenticated": False,
    "username": None,
    "user_role": None,
    "uploaded_vectorstore": None,
    "uploaded_documents": [],
    "processed_files": set(),
    "active_conversation_id": None,
}

for key, default_value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

# ------------------------------------------------------------
# LOGIN PAGE
# ------------------------------------------------------------

if not st.session_state.authenticated:
    st.title("🎓 EduSmart AI Assistant")
    st.subheader("Your AI-powered document and learning assistant")

    st.write(
        "Upload documents, ask questions, generate summaries, "
        "and explore learning materials."
    )

    with st.form("login_form"):
        username_input = st.selectbox(
            "Demo account",
            ["admin", "student", "employee"],
        )

        password_input = st.text_input(
            "Password",
            type="password",
        )

        submitted = st.form_submit_button(
            "Login",
            use_container_width=True,
        )

        if submitted:
            account = DEMO_USERS.get(username_input)

            if account and password_input == account["password"]:
                st.session_state.authenticated = True
                st.session_state.username = username_input
                st.session_state.user_role = account["role"]
                st.session_state.active_conversation_id = None
                st.rerun()
            else:
                st.error("Incorrect password. Please try again.")

    with st.expander("Demo credentials"):
        st.markdown(
            """
            - **Admin:** `admin` / `Admin123`
            - **Student:** `student` / `Student123`
            - **Employee:** `employee` / `Employee123`
            """
        )

    st.stop()

# ------------------------------------------------------------
# CURRENT USER
# ------------------------------------------------------------

username = st.session_state.username
role = st.session_state.user_role

# ------------------------------------------------------------
# SIDEBAR AND LOGOUT
# ------------------------------------------------------------

st.sidebar.title("EduSmart")
st.sidebar.success(f"User: {username}")
st.sidebar.caption(f"Role: {role}")

if st.sidebar.button("Logout", use_container_width=True):
    st.session_state.authenticated = False
    st.session_state.username = None
    st.session_state.user_role = None
    st.session_state.active_conversation_id = None

    # Keep uploaded documents available in this browser session.
    st.rerun()

st.sidebar.divider()

# ------------------------------------------------------------
# NAVIGATION OPTIONS
# ------------------------------------------------------------

if role == "Admin":
    modes = [
        "Ask EduSmart",
        "Upload Documents",
        "Take a Quiz",
        "Topic Summary",
        "Study Guide",
    ]
elif role == "Student":
    modes = [
        "Ask EduSmart",
        "Take a Quiz",
        "Topic Summary",
        "Study Guide",
    ]
else:
    modes = [
        "Ask EduSmart",
        "Topic Summary",
        "Study Guide",
    ]

# ------------------------------------------------------------
# CHAT HISTORY SIDEBAR
# ------------------------------------------------------------

st.sidebar.subheader("💬 Chat History")

try:
    conversations = get_conversations(username)
except Exception as exc:
    st.sidebar.error(f"Could not load chat history: {exc}")
    conversations = []

conversation_ids = {
    conversation["id"] for conversation in conversations
}

# Do not retain a selection belonging to another username.
if (
    st.session_state.active_conversation_id
    not in conversation_ids
):
    st.session_state.active_conversation_id = None

if st.sidebar.button(
    "➕ New Chat",
    use_container_width=True,
):
    st.session_state.active_conversation_id = None
    st.session_state["feature_navigation"] = "Ask EduSmart"
    st.rerun()

if conversations:
    for conversation in conversations:
        conversation_id = conversation["id"]
        title = conversation["title"]

        if len(title) > 32:
            title = title[:29] + "..."

        is_selected = (
            st.session_state.active_conversation_id
            == conversation_id
        )

        if st.sidebar.button(
            f"💬 {title}",
            key=f"history_{conversation_id}",
            type="primary" if is_selected else "secondary",
            use_container_width=True,
        ):
            st.session_state.active_conversation_id = conversation_id
            st.session_state["feature_navigation"] = "Ask EduSmart"
            st.rerun()
else:
    st.sidebar.caption("No saved conversations yet.")

st.sidebar.divider()

# Make sure the navigation selection is valid for this role.
if st.session_state.get("feature_navigation") not in modes:
    st.session_state["feature_navigation"] = modes[0]

mode = st.sidebar.radio(
    "Choose a feature",
    modes,
    key="feature_navigation",
)

# ------------------------------------------------------------
# DOCUMENT PROCESSING
# ------------------------------------------------------------

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

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )

    chunks = splitter.split_documents(new_documents)

    if not chunks:
        st.warning("No text chunks were created from these documents.")
        return

    all_documents = st.session_state.uploaded_documents + chunks

    try:
        with st.spinner("Creating searchable document index..."):
            vectorstore = FAISS.from_documents(
                all_documents,
                get_embeddings(),
            )

        st.session_state.uploaded_documents = all_documents
        st.session_state.uploaded_vectorstore = vectorstore
        st.session_state.processed_files.update(new_file_names)

        st.success(
            f"Processed {len(new_file_names)} file(s) successfully."
        )

    except Exception as exc:
        st.error(f"Could not create the document index: {exc}")


# ------------------------------------------------------------
# MAIN HEADER
# ------------------------------------------------------------

st.title("🎓 EduSmart AI Assistant")

st.caption(
    "AI-powered learning, document search and knowledge assistance"
)

st.write(
    f"Welcome, **{username}**! You are logged in as **{role}**."
)

# ------------------------------------------------------------
# ADMIN DOCUMENT UPLOAD
# ------------------------------------------------------------

if mode == "Upload Documents":
    st.header("📁 Document Library")

    st.write(
        "Upload files to extract their text and create a searchable "
        "knowledge base."
    )

    st.info(
        "Supported formats: PDF, DOCX, TXT, MD, CSV, XLSX and PPTX."
    )

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

# ------------------------------------------------------------
# ASK EDUSMART WITH PERSISTENT CHAT HISTORY
# ------------------------------------------------------------

elif mode == "Ask EduSmart":
    st.header("💬 Ask EduSmart")

    if st.session_state.uploaded_vectorstore is not None:
        st.success(
            "Uploaded document knowledge base is ready. "
            "Questions will use your uploaded documents."
        )
    else:
        st.info(
            "No uploaded document index is available in this session. "
            "Questions will use the existing curriculum knowledge base."
        )

    conversation_id = st.session_state.active_conversation_id

    # Display saved messages from the selected conversation.
    if conversation_id is not None:
        messages = get_messages(conversation_id)

        for message in messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Question form
    with st.form("ask_form", clear_on_submit=True):
        question = st.text_area(
            "Enter your question",
            placeholder=(
                "Example: Summarize this document or explain its key points."
            ),
            height=120,
            key="question_input",
        )

        submitted = st.form_submit_button(
            "Get Answer",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        if not question.strip():
            st.warning("Please enter a question.")

        else:
            conversation_id = (
                st.session_state.active_conversation_id
            )

            try:
                # Create a conversation for the first question.
                if conversation_id is None:
                    conversation_id = create_conversation(
                        username,
                        question.strip()[:100],
                    )

                    st.session_state.active_conversation_id = (
                        conversation_id
                    )

                # Save the user's question.
                save_message(
                    conversation_id,
                    "user",
                    question.strip(),
                )

                with st.spinner(
                    "EduSmart is preparing your answer..."
                ):
                    result = answer_question(
                        question.strip(),
                        vectorstore=(
                            st.session_state.uploaded_vectorstore
                        ),
                    )

                answer = str(result.get("answer", ""))

                # Store the answer together with its source details.
                answer_for_history = answer

                sources = result.get("sources", [])

                if sources:
                    answer_for_history += "\n\n**Sources**\n"
                    answer_for_history += "\n".join(
                        f"- {Path(str(source)).name}"
                        for source in sources
                    )

                documents = result.get("documents", [])

                if documents:
                    answer_for_history += (
                        "\n\n**Retrieved document excerpts**\n"
                    )

                    for index, document in enumerate(
                        documents,
                        start=1,
                    ):
                        metadata = document.metadata

                        answer_for_history += (
                            f"\n**Excerpt {index} — "
                            f"{metadata.get('source', 'Document')}**\n"
                        )

                        if "page" in metadata:
                            answer_for_history += (
                                f"Page: {metadata['page']}\n"
                            )

                        if "slide" in metadata:
                            answer_for_history += (
                                f"Slide: {metadata['slide']}\n"
                            )

                        if "sheet" in metadata:
                            answer_for_history += (
                                f"Worksheet: {metadata['sheet']}\n"
                            )

                        answer_for_history += (
                            "\n" + document.page_content + "\n"
                        )

                # Save the AI answer in SQLite.
                save_message(
                    conversation_id,
                    "assistant",
                    answer_for_history,
                )

                # Refresh the conversation and sidebar.
                st.rerun()

            except Exception as exc:
                st.error(f"Unable to answer your question: {exc}")

# ------------------------------------------------------------
# QUIZ
# ------------------------------------------------------------

elif mode == "Take a Quiz":
    st.header("📝 Take a Quiz")

    topic = st.text_input(
        "Enter a quiz topic",
        placeholder="Example: Photosynthesis",
    )

    if st.button("Generate Quiz", type="primary"):
        if not topic.strip():
            st.warning("Please enter a topic.")
        else:
            try:
                with st.spinner("Generating your quiz..."):
                    quiz_data = generate_quiz_data(topic.strip())

                st.subheader("Generated Quiz")

                if isinstance(quiz_data, (dict, list)):
                    st.json(quiz_data)
                else:
                    st.write(quiz_data)

            except Exception as exc:
                st.error(f"Could not generate quiz: {exc}")

# ------------------------------------------------------------
# TOPIC SUMMARY
# ------------------------------------------------------------

elif mode == "Topic Summary":
    st.header("📚 Topic Summary")

    topic = st.text_input(
        "Enter a topic to summarize",
        placeholder="Example: Photosynthesis",
    )

    if st.button("Generate Summary", type="primary"):
        if not topic.strip():
            st.warning("Please enter a topic.")
        else:
            try:
                with st.spinner("Preparing your summary..."):
                    result = run_agent(
                        f"Create a clear, structured summary of: "
                        f"{topic.strip()}. Include key concepts and "
                        f"important takeaways."
                    )

                st.subheader("Summary")

                if isinstance(result, str):
                    st.markdown(result)
                else:
                    st.write(result)

            except Exception as exc:
                st.error(f"Could not generate summary: {exc}")

# ------------------------------------------------------------
# STUDY GUIDE
# ------------------------------------------------------------

elif mode == "Study Guide":
    st.header("📖 Study Guide")

    topic = st.text_input(
        "Enter a topic for your study guide",
        placeholder="Example: The Human Respiratory System",
    )

    if st.button("Create Study Guide", type="primary"):
        if not topic.strip():
            st.warning("Please enter a topic.")
        else:
            try:
                prompt = (
                    f"Create a beginner-friendly study guide for "
                    f"{topic.strip()}. Include learning objectives, "
                    f"key concepts, examples, revision notes, and "
                    f"five practice questions."
                )

                with st.spinner("Creating your study guide..."):
                    result = run_agent(prompt)

                st.subheader("Your Study Guide")

                if isinstance(result, str):
                    st.markdown(result)
                else:
                    st.write(result)

            except Exception as exc:
                st.error(f"Could not create study guide: {exc}")

# ------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------

st.divider()

st.caption(
    "EduSmart AI Assistant | Demo application for college and office use"
)