
import argparse

import chromadb
import torch
from litellm import completion
from tqdm.auto import tqdm
from transformers import AutoModel




def main():
    parser = argparse.ArgumentParser(description="Responder preguntas de HotpotQA con RAG")
    parser.add_argument(
        "--collection",
        default="hotpot_qa",
        help="Nombre de la coleccion de pasajes en ChromaDB (default: hotpot_qa)",
    )
    parser.add_argument(
        "--chroma-path",
        default="./chroma",
        help="Directorio de la base de datos ChromaDB (default: ./chroma)",
    )
    parser.add_argument(
        "--embedding-model",
        default="jinaai/jina-embeddings-v5-text-nano",
        help="Modelo de embeddings (default: jinaai/jina-embeddings-v5-text-nano)",
    )
    parser.add_argument(
        "--model",
        default="gemini/gemini-3.1-flash-lite",
        help="Modelo LLM para litellm",
    )
    parser.add_argument(
        "--n-results",
        type=int,
        default=10,
        help="Cantidad de documentos a recuperar por pregunta (default: 10)",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limitar la cantidad de preguntas a responder"
    )
    args = parser.parse_args()

    client = chromadb.PersistentClient(args.chroma_path)
    ### BEGIN SOLUTION
    ### END SOLUTION


if __name__ == "__main__":
    main()
