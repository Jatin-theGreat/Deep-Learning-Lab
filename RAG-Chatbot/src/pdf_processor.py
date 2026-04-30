import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


UPLOAD_DIR = "uploads"


def save_uploaded_file(uploaded_file):
    """
    Save uploaded PDF to local uploads folder.
    """
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return file_path


def load_pdf(file_path):
    """
    Load a single PDF file.
    """
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents


def load_multiple_pdfs(file_paths):
    """
    Load multiple PDF files.
    """
    all_documents = []

    for file_path in file_paths:
        documents = load_pdf(file_path)
        all_documents.extend(documents)

    return all_documents


def split_documents(documents):
    """
    Split documents into smaller chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = text_splitter.split_documents(documents)
    return chunks