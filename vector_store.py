import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import config

def get_embeddings():
    return HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)

def get_vector_store():
    embeddings = get_embeddings()
    return Chroma(
        persist_directory=config.CHROMA_DB_DIR,
        embedding_function=embeddings,
        collection_name="insurance_docs"
    )

def add_documents(chunks):
    if not chunks:
        return
    vector_store = get_vector_store()
    vector_store.add_documents(chunks)

def is_document_ingested(source_name: str) -> bool:
    vector_store = get_vector_store()
    try:
        results = vector_store.get(where={"source": source_name}, limit=1)
        return len(results.get("ids", [])) > 0
    except Exception:
        return False