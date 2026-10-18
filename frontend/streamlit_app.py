
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

# ------------------------------------------------------------
# DATABASE AND PAGE CONFIG
# ------------------------------------------------------------

init_db()

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
    "generated_quiz": None,
    "quiz_version": 0,
    "quiz_submitted": False,
    "quiz_score": None,
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
        password_input = st.text_input("Password", type="password")

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
    st.rerun()

st.sidebar.divider()

# ------------------------------------------------------------
# NAVIGATION
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

if st.session_state.active_conversation_id not in conversation_ids:
    st.session_state.active_conversation_id = None

if st.sidebar.button("➕ New Chat", use_container_width=True):
    st.session_state.active_conversation_id = None
    st.session_state["feature_navigation"] = "Ask EduSmart"
    st.rerun()

for conversation in conversations:
    conversation_id = conversation["id"]
    title = conversation["title"]

    if len(title) > 32:
        title = title[:29] + "..."

    is_selected = (
        st.session_state.active_conversation_id == conversation_id
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

if not conversations:
    st.sidebar.caption("No saved conversations yet.")

st.sidebar.divider()

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
    """Extract, split and index uploaded documents."""

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
st.write(f"Welcome, **{username}**! You are logged in as **{role}**.")

# ------------------------------------------------------------
# ADMIN DOCUMENT UPLOAD
# ------------------------------------------------------------

if mode == "Upload Documents":
    st.header("📁 Document Library")

    st.write(
        "Upload files to extract their text and create a searchable "
        "knowledge base."
    )

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

# ------------------------------------------------------------
# ASK EDUSMART
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

    if conversation_id is not None:
        try:
            messages = get_messages(conversation_id)
            for message in messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
        except Exception as exc:
            st.error(f"Could not load messages: {exc}")

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
            conversation_id = st.session_state.active_conversation_id

            try:
                if conversation_id is None:
                    conversation_id = create_conversation(
                        username,
                        question.strip()[:100],
                    )
                    st.session_state.active_conversation_id = conversation_id

                save_message(conversation_id, "user", question.strip())

                with st.spinner("EduSmart is preparing your answer..."):
                    result = answer_question(
                        question.strip(),
                        vectorstore=st.session_state.uploaded_vectorstore,
                    )

                answer = str(result.get("answer", ""))
                answer_for_history = answer
                sources = result.get("sources", [])

                if sources:
                    answer_for_history += "\n\n**Sources**\n"
                    answer_for_history += "\n".join(
                        f"- {Path(str(source)).name}" for source in sources
                    )

                documents = result.get("documents", [])

                if documents:
                    answer_for_history += "\n\n**Retrieved document excerpts**\n"

                    for index, document in enumerate(documents, start=1):
                        metadata = document.metadata
                        source_name = Path(
                            str(metadata.get("source", "Document"))
                        ).name

                        answer_for_history += (
                            f"\n**Excerpt {index} — {source_name}**\n"
                        )

                        if "page" in metadata:
                            answer_for_history += f"Page: {metadata['page']}\n"
                        if "slide" in metadata:
                            answer_for_history += f"Slide: {metadata['slide']}\n"
                        if "sheet" in metadata:
                            answer_for_history += f"Worksheet: {metadata['sheet']}\n"

                        answer_for_history += "\n" + document.page_content + "\n"

                save_message(
                    conversation_id,
                    "assistant",
                    answer_for_history,
                )
                st.rerun()

            except Exception as exc:
                st.error(f"Unable to answer your question: {exc}")

# ------------------------------------------------------------
# TAKE A QUIZ
# ------------------------------------------------------------

elif mode == "Take a Quiz":
    st.header("📝 Take a Quiz")
    st.write("Generate a multiple-choice quiz and test your knowledge.")

    topic = st.text_input(
        "Enter a quiz topic",
        placeholder="Example: Photosynthesis",
        key="quiz_topic_input",
    )

    if st.button(
        "Generate Quiz",
        type="primary",
        key="generate_quiz_button",
    ):
        if not topic.strip():
            st.warning("Please enter a topic.")
        else:
            try:
                with st.spinner("Generating your quiz..."):
                    quiz_data = generate_quiz_data(topic.strip())

                if hasattr(quiz_data, "model_dump"):
                    quiz_data = quiz_data.model_dump()
                elif hasattr(quiz_data, "dict"):
                    quiz_data = quiz_data.dict()

                if not isinstance(quiz_data, dict):
                    raise ValueError(
                        "Unexpected quiz format. Expected a dictionary."
                    )

                questions = quiz_data.get("questions", [])

                if not questions:
                    raise ValueError("No questions were generated.")

                # Validate quiz structure before displaying it.
                for index, question in enumerate(questions, start=1):
                    if not isinstance(question, dict):
                        raise ValueError(
                            f"Question {index} is not a valid object."
                        )

                    options = question.get("options", [])
                    correct_answer = question.get("correct_answer")

                    if len(options) != 4:
                        raise ValueError(
                            f"Question {index} must have exactly four options."
                        )

                    if not all(
                        isinstance(option, str) and option.strip()
                        for option in options
                    ):
                        raise ValueError(
                            f"Question {index} has an invalid option."
                        )

                    if not any(
                        correct_answer.strip().casefold()
                        == option.strip().casefold()
                        for option in options
                    ) if isinstance(correct_answer, str) else True:
                        raise ValueError(
                            f"Question {index} has an invalid correct answer."
                        )

                    if not question.get("question", "").strip():
                        raise ValueError(
                            f"Question {index} has no question text."
                        )

                st.session_state.generated_quiz = quiz_data
                st.session_state.quiz_version += 1
                st.session_state.quiz_submitted = False
                st.session_state.quiz_score = None

            except Exception as exc:
                st.error(f"Could not generate quiz: {exc}")

    quiz_data = st.session_state.generated_quiz

    if quiz_data:
        questions = quiz_data["questions"]
        quiz_version = st.session_state.quiz_version

        st.divider()
        st.subheader("📝 Your Quiz")
        st.caption(f"Questions: {len(questions)}")

        with st.form(key=f"quiz_form_{quiz_version}"):
            for index, question in enumerate(questions, start=1):
                st.markdown(f"### Question {index}")
                st.write(question["question"])

                st.radio(
                    "Choose one answer:",
                    options=question["options"],
                    index=None,
                    key=f"quiz_{quiz_version}_answer_{index}",
                    label_visibility="collapsed",
                )
                st.divider()

            submitted = st.form_submit_button(
                "Submit Answers",
                type="primary",
                use_container_width=True,
            )

        if submitted:
            score = 0
            unanswered = 0
            results = []

            for index, question in enumerate(questions, start=1):
                selected_answer = st.session_state.get(
                    f"quiz_{quiz_version}_answer_{index}"
                )
                correct_answer = question["correct_answer"]

                is_correct = (
                    isinstance(selected_answer, str)
                    and selected_answer.strip().casefold()
                    == correct_answer.strip().casefold()
                )

                if selected_answer is None:
                    unanswered += 1

                if is_correct:
                    score += 1

                results.append({
                    "question_number": index,
                    "selected_answer": selected_answer,
                    "correct_answer": correct_answer,
                    "is_correct": is_correct,
                    "explanation": question.get(
                        "explanation",
                        "No explanation provided.",
                    ),
                })

            st.session_state.quiz_score = {
                "score": score,
                "total": len(questions),
                "unanswered": unanswered,
                "results": results,
            }
            st.session_state.quiz_submitted = True

        if st.session_state.quiz_submitted:
            result = st.session_state.quiz_score

            if result:
                score = result["score"]
                total = result["total"]
                percentage = round(score / total * 100)

                st.divider()
                st.subheader("📊 Your Results")

                col1, col2, col3 = st.columns(3)
                col1.metric("Score", f"{score}/{total}")
                col2.metric("Percentage", f"{percentage}%")
                col3.metric("Unanswered", result["unanswered"])

                if percentage >= 80:
                    st.success("Excellent work! 🎉")
                elif percentage >= 50:
                    st.info("Good effort! Review the explanations below.")
                else:
                    st.warning(
                        "Keep practising! Review the answers and try again."
                    )

                st.subheader("✅ Answer Review")

                for answer_result, question in zip(
                    result["results"], questions
                ):
                    number = answer_result["question_number"]
                    status = (
                        "Correct ✅"
                        if answer_result["is_correct"]
                        else "Incorrect ❌"
                    )

                    with st.expander(f"Question {number}: {status}"):
                        st.write(question["question"])
                        st.write(
                            "Your answer: "
                            f"{answer_result['selected_answer'] or 'Not answered'}"
                        )
                        st.write(
                            f"Correct answer: {answer_result['correct_answer']}"
                        )
                        st.write(
                            f"Explanation: {answer_result['explanation']}"
                        )

                if st.button(
                    "Generate Another Quiz",
                    type="primary",
                    key="another_quiz_button",
                ):
                    st.session_state.generated_quiz = None
                    st.session_state.quiz_submitted = False
                    st.session_state.quiz_score = None
                    st.rerun()

# ------------------------------------------------------------
# TOPIC SUMMARY
# ------------------------------------------------------------

elif mode == "Topic Summary":
    st.header("📚 Topic Summary")

    topic = st.text_input(
        "Enter a topic to summarize",
        placeholder="Example: Photosynthesis",
        key="summary_topic_input",
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
        key="study_guide_topic_input",
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
