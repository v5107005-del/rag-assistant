#!/usr/bin/env python3
"""Minimal Web/FastAPI interface on top of the existing rag-demo RAG pipeline.

Reuses ingest.py's vector store (chroma_db/) and query.py's answer() function
as-is — nothing about the retrieval/LLM logic is rewritten here, this file only
adds an HTTP interface (per Этап 3: "не переписывать систему; только добавить
Web/FastAPI-интерфейс").

Run locally (not exposed publicly — see rag-demo.md "Как запустить веб-демо"):
    uvicorn web_app:app --host 127.0.0.1 --port 8901
"""
from __future__ import annotations

import html
import time

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

from query import answer as rag_answer

app = FastAPI(title="RAG Assistant")

PAGE_TEMPLATE = """<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<title>RAG Assistant</title>
<style>
body {{ font-family: system-ui, sans-serif; max-width: 720px; margin: 40px auto; padding: 0 16px; color: #1a1a1a; }}
h1 {{ font-size: 1.3rem; }}
form {{ display: flex; gap: 8px; margin-bottom: 24px; }}
input[type=text] {{ flex: 1; padding: 8px; font-size: 1rem; }}
button {{ padding: 8px 16px; font-size: 1rem; }}
.answer-box {{ border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin-top: 16px; }}
.sources {{ color: #555; font-size: 0.9rem; margin-top: 8px; }}
.meta {{ color: #888; font-size: 0.8rem; }}
</style>
</head>
<body>
<h1>RAG Assistant — вопросы по документации</h1>
<p>ChromaDB (локально) + OpenAI embeddings/LLM. По умолчанию база знаний — README открытых проектов автора (knowledge_base/), можно заменить на свои документы.</p>
<form method="get" action="/ask">
  <input type="text" name="q" placeholder="Например: Как происходит публикация статьи?" value="{question_escaped}">
  <button type="submit">Спросить</button>
</form>
{result_block}
</body>
</html>"""

RESULT_TEMPLATE = """<div class="answer-box">
  <strong>Вопрос:</strong> {question}<br><br>
  <strong>Ответ:</strong><br>{answer}
  <div class="sources"><strong>Источники:</strong> {sources}</div>
  <div class="meta">Время ответа: {elapsed}s · чанков в контексте: {chunks}</div>
</div>"""


def render_page(question: str | None = None, result: dict | None = None) -> str:
    question_escaped = html.escape(question or "")
    result_block = ""
    if result:
        result_block = RESULT_TEMPLATE.format(
            question=html.escape(result["question"]),
            answer=html.escape(result["answer"]).replace("\n", "<br>"),
            sources=html.escape(", ".join(result["sources"])) or "—",
            elapsed=result["elapsed_seconds"],
            chunks=result["retrieved_chunks"],
        )
    return PAGE_TEMPLATE.format(question_escaped=question_escaped, result_block=result_block)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return render_page()


@app.get("/ask", response_class=HTMLResponse)
def ask(q: str) -> str:
    result = rag_answer(q)
    return render_page(question=q, result=result)


@app.get("/api/ask", response_class=JSONResponse)
def ask_api(q: str) -> dict:
    return rag_answer(q)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "time": time.time()}
