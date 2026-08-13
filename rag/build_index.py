"""
One-time (re-runnable) script: reads rag/faq_data.md, splits it into one
chunk per Q&A entry (split on '## ' headings — natural, meaning-complete
units, not fixed-token windows), embeds each chunk, and persists them into
a local Chroma index at rag/vectorstore/.

Run from rag/:
    python build_index.py

Requires OPENAI_API_KEY set as an environment variable.
Re-run this any time faq_data.md changes — it rebuilds the index from scratch.
"""
import os
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

FAQ_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "faq_data.md")
PERSIST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vectorstore")


def parse_faq_md(path: str) -> list[Document]:
    """
    Splits the FAQ markdown into one Document per '## Question' section.
    Each Document's page_content is "Question: ...\nAnswer: ..." so the
    embedded text carries the question itself, not just the answer prose
    (helps retrieval match on question-style caller queries).
    """
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # split on '## ' headings, first chunk (before any heading) is the title line — drop it
    raw_sections = content.split("\n## ")
    sections = raw_sections[1:]  # drop the "# Health1st Clinic — FAQ" title block

    documents = []
    for section in sections:
        lines = section.strip().split("\n", 1)
        question = lines[0].strip()
        answer = lines[1].strip() if len(lines) > 1 else ""

        documents.append(
            Document(
                page_content=f"Question: {question}\nAnswer: {answer}",
                metadata={"question": question},
            )
        )
    return documents


def build_index():
    documents = parse_faq_md(FAQ_PATH)
    print(f"Parsed {len(documents)} FAQ entries from {FAQ_PATH}")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Fresh build every time — wipe existing persisted dir first for a clean rebuild
    if os.path.exists(PERSIST_DIR):
        import shutil
        shutil.rmtree(PERSIST_DIR)

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
    )
    print(f"Built and persisted vector index at {PERSIST_DIR}")
    return vectorstore


if __name__ == "__main__":
    build_index()
