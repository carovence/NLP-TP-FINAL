import argparse
import chromadb
import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoModel




def main():
    parser = argparse.ArgumentParser(description="Ingestar HotpotQA en ChromaDB")
    parser.add_argument(
        "--dataset",
        default="nlp-udesa/hotpot_qa_3k",
        help="Nombre del dataset en HuggingFace (default: nlp-udesa/hotpot_qa_3k)",
    )
    parser.add_argument(
        "--embedding-model",
        default="jinaai/jina-embeddings-v5-text-nano",
        help="Modelo de embeddings",
    )
    parser.add_argument(
        "--collection",
        default="hotpot_qa",
        help="Nombre de la coleccion en ChromaDB (default: hotpot_qa)",
    )
    parser.add_argument(
        "--chroma-path",
        default="./chroma",
        help="Directorio para la base de datos ChromaDB",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Tamaño de batch para embeddings (default: 96)",
    )
    parser.add_argument(
        "--model-max-length",
        type=int,
        default=4096,
        help="Longitud máxima del tokenizador (default: 4096)",
    )
    args = parser.parse_args()

    ### BEGIN SOLUTION
    # Cargar dataset

    # Cargar modelo de embeddings

    # Guardar todos los pasajes en ChromaDB

    # Crear coleccion de preguntas para ir guardando respuestas...

if __name__ == "__main__":
    main()
