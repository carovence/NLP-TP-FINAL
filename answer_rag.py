import argparse
import json
import os

import torch
import chromadb
from datasets import load_dataset
from dotenv import load_dotenv
from litellm import completion
from transformers import AutoModel, AutoTokenizer

# carga las variables del archivo .env (entre ellas OPENAI_API_KEY)
load_dotenv()


def armar_prompt(pregunta, contextos):
    contexto = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contextos))
    return (
        "Sos un sistema de extraccion de respuestas para el dataset HotpotQA,"
        "un dataset de question answering multi-hop en inglés."
        "Usas los pasajes provistos, ordenados de mas a menos relevante ([1] es el mas similar a la pregunta).\n\n"
        "La respuesta siempre está en los pasajes (se pueden llegar a combinar varios pasajes para una misma respuesta).\n\n"
        "Reglas de formato:\n"
        "- Respondé en inglés.\n"
        "- Respondé solo con la respuesta, lo mas corta posible: una palabra, un nombre, "
        "una fecha, un numero o una frase breve.\n"
        "- No escribas oraciones completas. \n"
        "- No repitas la pregunta.\n"
        "- No expliques ni justifiques.\n"
        "- Si la respuesta aparece literal en un pasaje, copiá exactamente ese fragmento (el minimo necesario).\n"
        "- Para preguntas de si/no respondé exactamente 'yes' o 'no', en minuscula.\n"
        "- Nunca digas 'I don't know' ni 'not enough information'. Siempre dá tu mejor respuesta.\n"
        "- Si dudas, basate en los pasajes de numero mas bajo (los mas relevantes).\n"
        "- Identificá que tipo de dato pide la pregunta (un largo/distancia, una fecha, "
        "una cantidad, una persona, un lugar, un titulo) y devolvé exactamente ese tipo, "
        "no un dato relacionado.\n\n"
        f"PASAJES:\n{contexto}\n\n"
        f"PREGUNTA: {pregunta}\nRESPUESTA:"
    )


def main():
    parser = argparse.ArgumentParser(description="Responder preguntas de HotpotQA con RAG")
    parser.add_argument(
        "--dataset",
        default="nlp-udesa/hotpot_qa_3k",
        help="Dataset de HuggingFace (default: nlp-udesa/hotpot_qa_3k)",
    )
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
        default="gpt-4o-mini",
        help="Modelo LLM para litellm (usa OPENAI_API_KEY del entorno)",
    )
    parser.add_argument(
        "--n-results",
        type=int,
        default=40,
        help="Cantidad de documentos a recuperar por pregunta (default: 20)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Limitar la cantidad de preguntas a responder (default: 10 para prueba)",
    )
    parser.add_argument(
        "--output",
        default="respuestas_rag.jsonl",
        help="Archivo de salida (default: respuestas_rag.jsonl)",
    )
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit(
            "Falta OPENAI_API_KEY. Agregala en un archivo .env "
            "(OPENAI_API_KEY=tu-key) o exportala en la terminal."
        )

    ### BEGIN SOLUTION

    # dataset
    ds = load_dataset(args.dataset)["validation"]
    if args.limit is not None:
        ds = ds.select(range(args.limit))

    # modelo de embeddings
    tokenizer = AutoTokenizer.from_pretrained(args.embedding_model)
    model = AutoModel.from_pretrained(args.embedding_model)
    model.eval()

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    model = model.to(device=device)

    # conectar a ChromaDB
    client = chromadb.PersistentClient(args.chroma_path)
    collection = client.get_collection(args.collection)

    instruction = "Represent this sentence for searching relevant passages: "
    respuestas = []

    for ejemplo in ds:
        pregunta = ejemplo["question"]

        # embedding de la pregunta 
        encoded = tokenizer(
            [instruction + pregunta],
            padding=True, truncation=True, return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            output = model(**encoded)
            embedding = output[0][:, 0]
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)

        # buscar top-k en chroma
        results = collection.query(
            query_embeddings=embedding[0].cpu().numpy(),
            n_results=args.n_results,
            include=["documents"],
        )
        contextos = results["documents"][0]

        # armar prompt y llamar al LLM
        prompt = armar_prompt(pregunta, contextos)
        response = completion(
            model=args.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        respuesta_pred = response.choices[0].message.content

        respuestas.append({
            "id": ejemplo["id"],
            "question": pregunta,
            "answer_pred": respuesta_pred,
            "answer_true": ejemplo["answer"],
        })

    # exportar al final
    with open(args.output, "w") as f:
        for r in respuestas:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Listo. {len(respuestas)} respuestas guardadas en {args.output}.")

    ### END SOLUTION


if __name__ == "__main__":
    main()
