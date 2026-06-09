"""
Paso 1 del pipeline RAG: precomputa los embeddings de las preguntas del split
validation y los guarda a un JSONL que despues consume answer_rag.py.

Corrélo manualmente desde la raiz del proyecto:

    python esqueleto/src/hotpot_qa/_embed_questions.py

Defaults: dataset = nlp-udesa/hotpot_qa_3k, limit = 10, modelo = BAAI/bge-small-en-v1.5,
output = ./query_embeddings.jsonl
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import json

import torch
torch.set_num_threads(1)

from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Precomputar embeddings de queries para RAG")
    parser.add_argument("--dataset", default="nlp-udesa/hotpot_qa_3k")
    parser.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--output", default="query_embeddings.jsonl")
    args = parser.parse_args()

    ds = load_dataset(args.dataset)["validation"]
    if args.limit is not None:
        ds = ds.select(range(args.limit))

    tokenizer = AutoTokenizer.from_pretrained(args.embedding_model)
    model = AutoModel.from_pretrained(args.embedding_model)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device=device)

    instruction = "Represent this sentence for searching relevant passages: "
    print(f"Computando embeddings para {len(ds)} preguntas con {args.embedding_model}...")

    with open(args.output, "w") as f:
        for ejemplo in tqdm(ds):
            enc = tokenizer(
                [instruction + ejemplo["question"]],
                padding=True, truncation=True, return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                out = model(**enc)
                emb = out[0][:, 0]
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            f.write(json.dumps({
                "id": ejemplo["id"],
                "question": ejemplo["question"],
                "answer_true": ejemplo["answer"],
                "embedding": emb[0].cpu().numpy().tolist(),
            }, ensure_ascii=False) + "\n")

    print(f"\nListo. Embeddings escritos a {args.output}. Ahora corré answer_rag.py.")


if __name__ == "__main__":
    main()
