from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS


BASE_DIR = Path(__file__).resolve().parent
CURRICULUM_DIR = BASE_DIR / "curriculum"
FAISS_DIR = BASE_DIR / "faiss_index"


def main():
    print("Starting EduSmart ingestion...")

    pdf_files = sorted(CURRICULUM_DIR.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in: {CURRICULUM_DIR}")
        return

    print(f"Found {len(pdf_files)} PDF files.")

    documents = []
    failed_files = []

    for pdf_file in pdf_files:
        print(f"\nLoading: {pdf_file.name}")

        try:
            loader = PyPDFLoader(str(pdf_file))
            docs = loader.load()

            documents.extend(docs)

            print(f"  Loaded {len(docs)} pages.")

        except Exception as e:
            print(f"  WARNING: Could not load {pdf_file.name}")
            print(f"  Reason: {type(e).__name__}: {e}")

            failed_files.append(pdf_file.name)

    if not documents:
        raise RuntimeError(
            "No PDF documents could be loaded. "
            "Please check the curriculum PDFs."
        )

    print(f"\nTotal pages loaded: {len(documents)}")

    
# Split documents into smaller, more focused chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=120,
        add_start_index=True,
    )


    chunks = text_splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks.")

    # Create embeddings
    print("\nCreating embeddings...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Create FAISS
    print("Building FAISS index...")

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings,
    )

    # Save
    FAISS_DIR.mkdir(parents=True, exist_ok=True)

    vectorstore.save_local(str(FAISS_DIR))

    print("\n===================================")
    print("FAISS index created successfully!")
    print(f"Location: {FAISS_DIR}")
    print(f"Total vectors: {vectorstore.index.ntotal}")
    print("===================================")

    if failed_files:
        print("\nWARNING: These files could not be processed:")
        for filename in failed_files:
            print(f"  - {filename}")

        print(
            "\nThe remaining curriculum files were successfully "
            "indexed and can be used by EduSmart."
        )


if __name__ == "__main__":
    main()