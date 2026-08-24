#!/usr/bin/env python3
"""Minimal RAG ingestion: Document Loader -> Text Splitter -> Embeddings -> Vector Store (Chroma, local).

Knowledge base: README files of the author's own public AI-agent-infrastructure
repositories (see knowledge_base/README.md for the list and sources) — drop any
Markdown/HTML files into knowledge_base/ to index your own documents instead.
No external vector service is used: Chroma runs as a local persistent client.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
import chromadb

BASE = Path(__file__).resolve().parent
KB_DIR = BASE / "knowledge_base"
CHROMA_DIR = BASE / "chroma_db"
COLLECTION_NAME = "docs_kb"
EMBEDDING_MODEL = "text-embedding-3-small"

load_dotenv(BASE / ".env")


def load_document(path: Path) -> str:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    if path.suffix.lower() == ".html":
        soup = BeautifulSoup(raw, "html.parser")
        return soup.get_text(separator="\n")
    return raw


def load_all_documents() -> list[dict]:
    docs = []
    for path in sorted(KB_DIR.iterdir()):
        if not path.is_file() or path.name == "README.md":
            continue
        text = load_document(path)
        docs.append({"source": path.name, "text": text})
    return docs


def split_documents(docs: list[dict]) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=150,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " "],
    )
    chunks = []
    for doc in docs:
        pieces = splitter.split_text(doc["text"])
        for i, piece in enumerate(pieces):
            chunks.append({
                "id": f"{doc['source']}::chunk{i}",
                "text": piece,
                "source": doc["source"],
            })
    return chunks


def embed_texts(client: OpenAI, texts: list[str]) -> list[list[float]]:
    # Batch to stay well under request-size limits.
    vectors: list[list[float]] = []
    batch_size = 64
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        vectors.extend([item.embedding for item in resp.data])
    return vectors


def main() -> None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    t0 = time.time()
    docs = load_all_documents()
    print(f"[loader] loaded {len(docs)} documents from {KB_DIR}")

    chunks = split_documents(docs)
    print(f"[splitter] produced {len(chunks)} chunks (chunk_size=1200, overlap=150)")

    client = OpenAI(api_key=api_key)
    texts = [c["text"] for c in chunks]
    vectors = embed_texts(client, texts)
    print(f"[embeddings] computed {len(vectors)} vectors with {EMBEDDING_MODEL}")

    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        chroma_client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = chroma_client.create_collection(COLLECTION_NAME)

    collection.add(
        ids=[c["id"] for c in chunks],
        embeddings=vectors,
        documents=texts,
        metadatas=[{"source": c["source"]} for c in chunks],
    )

    elapsed = time.time() - t0
    print(f"[vector_store] persisted {len(chunks)} chunks to {CHROMA_DIR} (collection={COLLECTION_NAME})")
    print(f"[done] total ingest time: {elapsed:.1f}s")

    stats = {
        "documents": len(docs),
        "chunks": len(chunks),
        "embedding_model": EMBEDDING_MODEL,
        "vector_store": "chromadb (local, persistent)",
        "ingest_seconds": round(elapsed, 1),
    }
    (BASE / "ingest_stats.json").write_text(
        __import__("json").dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
