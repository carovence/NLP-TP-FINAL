# -*- coding: utf-8 -*-
"""
NLP UdeSA 2026 — 09: Vector DB + LiteLLM
Materia: Procesamiento de Lenguaje Natural
Maestría en Ciencia de Datos — UdeSA, 2026
Profesores: Juan Manuel Pérez, Bruno Bianchi

Para correrlo:
    1) Instalar dependencias (en la terminal, NO acá adentro):
        /opt/homebrew/bin/python3.13 -m pip install chromadb torchao==0.16.0 transformers torch litellm

    2) Exportar la API key de Gemini (https://ai.google.dev/gemini-api/docs/api-key):
        export GEMINI_API_KEY=tu_api_key_aca

    3) Ejecutar:
        /opt/homebrew/bin/python3.13 ChromaDB.py
"""

import os
import sys

import chromadb
import torch
from transformers import AutoModel
from litellm import completion


# ---------------------------------------------------------------------------
# 1) ChromaDB con embeddings automáticos
# ---------------------------------------------------------------------------
print("\n=== 1) ChromaDB con embeddings automáticos ===")

client = chromadb.PersistentClient("./chroma")
collection = client.get_or_create_collection("demo")

textos = [
    "Python es un lenguaje de programación muy usado en aprendizaje supervisado.",
    "Vaaaaamooo boka",
    "A veces dos y dos son cinco.",
]

collection.upsert(
    ids=["1", "2", "3"],
    documents=textos,
)

print(f"Documentos en la colección: {collection.count()}")

query = "¿Qué herramientas se usan para machine learning?"
results = collection.query(
    query_texts=[query],
    n_results=2,
    include=["documents", "distances", "metadatas"],
)
print(results)


# ---------------------------------------------------------------------------
# 2) ChromaDB con embeddings propios (Jina)
# ---------------------------------------------------------------------------
print("\n=== 2) ChromaDB con embeddings propios (Jina) ===")

model_name = "jinaai/jina-embeddings-v5-text-nano"
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"Usando device: {device}")

model = AutoModel.from_pretrained(
    model_name,
    trust_remote_code=True,
    dtype=torch.bfloat16,
).to(device=device)

collection = client.get_or_create_collection("demo_embeddings")
embeddings = model.encode(textos, task="retrieval", prompt_name="document")

collection.upsert(
    ids=["1", "2", "3"],
    documents=textos,
    embeddings=embeddings.cpu().float().numpy(),
)

query_embedding = model.encode([query], task="retrieval", prompt_name="query")
results = collection.query(
    query_embeddings=query_embedding[0].cpu().float().numpy(),
    n_results=2,
    include=["documents", "distances", "metadatas"],
)
print(results)


# ---------------------------------------------------------------------------
# 3) LiteLLM (llamada a Gemini)
# ---------------------------------------------------------------------------
print("\n=== 3) LiteLLM (Gemini) ===")

if not os.environ.get("GEMINI_API_KEY"):
    print("Saltando llamada a Gemini: falta la variable de entorno GEMINI_API_KEY.")
    print("Exportala con:  export GEMINI_API_KEY=tu_api_key_aca")
    sys.exit(0)

response = completion(
    model="gemini/gemini-2.0-flash",
    messages=[{"role": "user", "content": "¿Qué es NLP? Respondé en una oración."}],
)
print(response.choices[0].message.content)
