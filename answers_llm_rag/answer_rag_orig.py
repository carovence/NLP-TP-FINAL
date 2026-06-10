import argparse
import string
from collections import Counter

from dotenv import load_dotenv
load_dotenv()

import chromadb
import torch
from datasets import load_dataset
from litellm import completion
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel




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
        default="BAAI/bge-small-en-v1.5",
        help="Modelo de embeddings (default: BAAI/bge-small-en-v1.5)",
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