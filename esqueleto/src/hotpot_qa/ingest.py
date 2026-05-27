#%%
import argparse
from types import SimpleNamespace
import chromadb
import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModel
from transformers.dynamic_module_utils import get_class_from_dynamic_module


#%%
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
print('SPLIT:', 'validation')
print('CAMPOS:', ds[split].column_names)
print('EJEMPLO:')
print(ds[split][0])

#%%
ds.keys()
len(ds[split])

#%%

def main():
    parser = argparse.ArgumentParser(description="Ingestar HotpotQA en ChromaDB")
    parser.add_argument(
        "--dataset",
        default="nlp-udesa/hotpot_qa_3k",
        help="Nombre del dataset en HuggingFace (default: nlp-udesa/hotpot_qa_3k)",
    )
    parser.add_argument(
        "--embedding-model",-
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
# Cargar dataset

ds = load_dataset(args.dataset)
#split = 'validation' 
ds = ds['validation'] 
   

#%% 
'''import pandas as pd

dataset_new = ds.to_pandas()
dataset_new.head()'''

#%%
MODEL_ID = "jinaai/jina-embeddings-v5-text-nano"

config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
model_class = get_class_from_dynamic_module(config.auto_map["AutoModel"], MODEL_ID)

model = model_class.from_pretrained(
    MODEL_ID,
    config=config,
    dtype=torch.bfloat16,
    trust_remote_code=True,
)

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
model = model.to(device=device)

#%%

ejemplo = ds[0]
sentences = ejemplo['context']['sentences']   # lista de parrafos (cada uno lista de oraciones)

print('Cantidad de parrafos:', len(sentences))
print()

#%%
# loop para unir oraciones
textos = []
for parrafo in sentences:
    # parrafo es una lista de oraciones, ej: ['or1', 'or2']
    # uni las oraciones en UN string con join
    texto_unido = ' '.join(parrafo)    # <-- ESTA es la linea clave
    textos.append(texto_unido)

for i in range(2):
    print(f'--- Texto {i} ---')
    print(textos[i][:200])
    print()
"
#%%
collection = client.get_or_create_collection("demo_embeddings")
embeddings = model.encode(ds[split]["context"], task="retrieval", prompt_name="document")


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
# Crear cliente ChromaDB


#%%
    # Cargar dataset



    ### END SOLUTION

if __name__ == "__main__":
    main()
