"""
Loads the persisted Chroma index (built by build_index.py) and exposes a
search function. Imported into langgraph_app/tools.py as the
search_clinic_faq tool.
"""
import os
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

PERSIST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vectorstore")

_vectorstore = None  # lazy-loaded singleton, avoid reloading on every call


def _get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        _vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
    return _vectorstore


def search_clinic_faq(query: str, k: int = 2) -> dict:
    """
    Searches the clinic FAQ knowledge base for content relevant to the query.
    Returns the top-k matching Q&A entries as grounding context.
    """
    if not os.path.exists(PERSIST_DIR):
        return {"error": "FAQ index not built yet. Run rag/build_index.py first."}

    vectorstore = _get_vectorstore()
    results = vectorstore.similarity_search(query, k=k)

    return {
        "results": [
            {"question": r.metadata.get("question", ""), "content": r.page_content}
            for r in results
        ]
    }
