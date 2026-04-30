import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

CHROMA_DB_DIR = "chroma_db"


def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def create_vector_store(chunks):
    os.makedirs(CHROMA_DB_DIR, exist_ok=True)
    embeddings = get_embedding_model()
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR
    )
    return vector_store


def load_vector_store():
    embeddings = get_embedding_model()
    return Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embeddings
    )