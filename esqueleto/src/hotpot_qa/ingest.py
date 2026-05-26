#%%
import argparse
from types import SimpleNamespace

import chromadb
import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoModel

# Defaults para poder correr las celdas interactivas sin pasar por argparse.
# Cuando el script se corre como CLI (python ingest.py ...), main() crea
# su propio `args` adentro y este se ignora.
args = SimpleNamespace(
    dataset="nlp-udesa/hotpot_qa_3k",
    embedding_model="jinaai/jina-embeddings-v5-text-nano",
    collection="hotpot_qa",
    chroma_path="./chroma",
    batch_size=8,
    model_max_length=4096,
)
#%%


ds = load_dataset(args.dataset)
print(ds)
split = list(ds.keys())[0]
print('SPLIT:', split)
print('CAMPOS:', ds[split].column_names)
print('EJEMPLO:')
print(ds[split][0])


#%%

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
#%%

### BEGIN SOLUTION

    # 1. Cargar dataset
    ds = load_dataset('nlp-udesa/hotpot_qa_3k')
    split = "validation"
#%%

    # 2. Cargar modelo de embeddings (Jina) 
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    model = AutoModel.from_pretrained(
        args.embedding_model,
        trust_remote_code=True,
        dtype=torch.bfloat16,
    ).to(device=device)
#%%
    # 3. Extraer y deduplicar pasajes (title + sentences)
    pasajes = {}  
    for ejemplo in ds[split]:
        titulos = ejemplo["context"]["title"]
        oraciones = ejemplo["context"]["sentences"]
        for titulo, ors in zip(titulos, oraciones):
            if titulo not in pasajes:
                pasajes[titulo] = " ".join(ors)

    titulos = list(pasajes.keys())
    textos = list(pasajes.values())
    print(f"Total de pasajes unicos: {len(textos)}")

    # 4. Conectar a ChromaDB y crear la coleccion - notebook 09
    client = chromadb.PersistentClient(args.chroma_path)
    collection = client.get_or_create_collection(args.collection)

    # 5. Vectorizar con Jina (encode) y guardar en ChromaDB, en lotes
    for i in tqdm(range(0, len(textos), args.batch_size)):
        batch_textos = textos[i : i + args.batch_size]
        batch_titulos = titulos[i : i + len(batch_textos)]
        batch_ids = [str(j) for j in range(i, i + len(batch_textos))]

        # vectorizar con Jina (notebook 08) - prompt_name="document"
        embeddings = model.encode(batch_textos, task="retrieval", prompt_name="document")

        # guardar en ChromaDB (notebook 09)
        collection.add(
            ids=batch_ids,
            documents=batch_textos,
            embeddings=embeddings.cpu().numpy(),
            metadatas=[{"titulo": t} for t in batch_titulos],
        )

    print(f"Listo. Documentos en la coleccion: {collection.count()}")

    ### END SOLUTION

if __name__ == "__main__":
    main()
