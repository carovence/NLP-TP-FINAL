#%%
import argparse
import chromadb
import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel

def main():
    parser = argparse.ArgumentParser(description="Ingestar HotpotQA en ChromaDB")
    parser.add_argument(
        "--dataset",
        default="nlp-udesa/hotpot_qa_3k",
        help="Nombre del dataset en HuggingFace (default: nlp-udesa/hotpot_qa_3k)",
    )
    parser.add_argument(
        "--embedding-model",
        default="BAAI/bge-small-en-v1.5",
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
        default=512,
        help="Longitud máxima del tokenizador (default: 512)",
    )
    args = parser.parse_args()

    # === 1. Cargar dataset ===
    ds = load_dataset(args.dataset)
    ds = ds["validation"]

    # === 2. Extraer y deduplicar pasajes ===
    corpus = {}   # diccionario: titulo -> texto
    for respuesta in ds:
        titulo_rta = respuesta["context"]["title"]
        oraciones_rta = respuesta["context"]["sentences"]
        for titulo, parrafo in zip(titulo_rta, oraciones_rta):
            if titulo in corpus:
                continue
            texto_oraciones = " ".join(parrafo)
            corpus[titulo] = texto_oraciones

    titulos = list(corpus.keys())
    textos = list(corpus.values())

    # === 3. Cargar modelo BGE ===
    tokenizer = AutoTokenizer.from_pretrained(args.embedding_model)
    model = AutoModel.from_pretrained(args.embedding_model)
    model.eval()

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    model = model.to(device=device)

    # === 4. Conectar a ChromaDB ===
    client = chromadb.PersistentClient(args.chroma_path)
    collection = client.get_or_create_collection(args.collection)

    # === 5. Vectorizar en lotes y guardar ===
    for i in tqdm(range(0, len(textos), args.batch_size)):
        batch_textos = textos[i : i + args.batch_size]
        batch_titulos = titulos[i : i + len(batch_textos)]
        batch_ids = [str(j) for j in range(i, i + len(batch_textos))]

        encoded_input = tokenizer(
            batch_textos,
            padding=True,
            truncation=True,
            max_length=args.model_max_length,
            return_tensors='pt',
        ).to(device)

        with torch.no_grad():
            model_output = model(**encoded_input)
            sentence_embeddings = model_output[0][:, 0]   # CLS pooling

        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)

        collection.add(
            ids=batch_ids,
            documents=batch_textos,
            embeddings=sentence_embeddings.cpu().numpy(),
            metadatas=[{"titulo": t} for t in batch_titulos],
        )

    print(f"Listo. Total de documentos en ChromaDB: {collection.count()}")

if __name__ == "__main__":
    main()





