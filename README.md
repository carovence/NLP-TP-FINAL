# HotpotQA TP

Trabajo practico de Question Answering multi-hop usando el dataset [HotpotQA](https://hotpotqa.github.io/).

## Setup

0. Instalar uv

Mirar en la página de [uv](https://docs.astral.sh/uv/getting-started/installation/). En Linux/MacOS/WSL se puede usar:

```
curl -LsSf https://astral.sh/uv/install.sh | sh
````

1. Instalar dependencias

```bash
uv sync
```

## Ingesta del dataset

Para cargar los pasajes del dataset `nlp-udesa/hotpot_qa_3k` en una base de datos ChromaDB local:

```bash
uv run ingest
```

Esto como paso previo a responder las preguntas con alguno de los tres modelos siguientes: LLM directo, RAG o agente.

## Responder preguntas

Para responder preguntas usando el modelo RAG:

```bash
uv run answer_direct # Responde usando solo el modelo de lenguaje, sin recuperar documentos.
uv run answer_rag    # Responde usando el modelo RAG, recuperando documentos relevantes.
uv run answer_agent  # Responde usando un agente
```
