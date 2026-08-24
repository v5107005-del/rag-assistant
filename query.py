#!/usr/bin/env python3
"""Minimal RAG query: Retriever -> LLM -> answer with sources.

Usage:
    python3 query.py "Как происходит публикация статьи?"
    python3 query.py --json "..."   # machine-readable output for the test harness
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
import chromadb

BASE = Path(__file__).resolve().parent
CHROMA_DIR = BASE / "chroma_db"
COLLECTION_NAME = "docs_kb"
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
TOP_K = 4

load_dotenv(BASE / ".env")

SYSTEM_PROMPT = """Ты — ассистент, отвечающий на вопросы по документации из базы знаний \
(README проектов из папки knowledge_base/). Отвечай ТОЛЬКО на основе предоставленного \
контекста. Если в контексте нет ответа на вопрос — прямо и честно скажи, что в базе \
знаний этой информации нет. Не придумывай факты, которых нет в контексте. Отвечай кратко \
и по-русски."""


def retrieve(client: OpenAI, collection, question: str, top_k: int = TOP_K):
    q_emb = client.embeddings.create(model=EMBEDDING_MODEL, input=[question]).data[0].embedding
    result = collection.query(query_embeddings=[q_emb], n_results=top_k)
    hits = []
    for doc, meta, dist in zip(result["documents"][0], result["metadatas"][0], result["distances"][0]):
        hits.append({"text": doc, "source": meta["source"], "distance": dist})
    return hits


def answer(question: str) -> dict:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    client = OpenAI(api_key=api_key)
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = chroma_client.get_collection(COLLECTION_NAME)

    t0 = time.time()
    hits = retrieve(client, collection, question)
    context = "\n\n---\n\n".join(f"[Источник: {h['source']}]\n{h['text']}" for h in hits)

    completion = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Контекст из базы знаний:\n\n{context}\n\nВопрос: {question}"},
        ],
        temperature=0.1,
        max_tokens=500,
    )
    elapsed = time.time() - t0

    sources = sorted({h["source"] for h in hits})
    return {
        "question": question,
        "answer": completion.choices[0].message.content,
        "sources": sources,
        "elapsed_seconds": round(elapsed, 2),
        "retrieved_chunks": len(hits),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    result = answer(args.question)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"Вопрос: {result['question']}\n")
        print(f"Ответ:\n{result['answer']}\n")
        print(f"Источники: {', '.join(result['sources'])}")
        print(f"Время ответа: {result['elapsed_seconds']}s")


if __name__ == "__main__":
    main()
