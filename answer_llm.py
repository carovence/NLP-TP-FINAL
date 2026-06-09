import argparse
import json
import os

from datasets import load_dataset
from dotenv import load_dotenv
from litellm import completion

# carga las variables del archivo .env (entre ellas OPENAI_API_KEY)
load_dotenv()


def armar_prompt(pregunta):
    return (
        "Sos un sistema de question answering para HotpotQA, "
        "un dataset de preguntas MULTI-HOP en inglés."
        "Reglas de formato:\n"
        "- Respondé en inglés.\n"
        "- Respondé solo con la respuesta, lo mas corta posible: una palabra, un nombre, "
        "una fecha, un numero o una frase breve.\n"
        "- No escribas oraciones completas.\n"
        "- No repitas la pregunta.\n"
        "- No expliques ni justifiques.\n"
        "- Para preguntas de si/no respondé exactamente 'yes' o 'no', en minuscula.\n"
        "- Nunca digas 'I don't know' ni 'not enough information'. Siempre dá tu mejor respuesta.\n"
        "- Identificá que tipo de dato pide la pregunta (un largo/distancia, una fecha, "
        "una cantidad, una persona, un lugar, un titulo) y devolvé exactamente ese tipo, "
        "no un dato relacionado. \n\n"
        f"PREGUNTA: {pregunta}\nRESPUESTA:"
    )


def main():
    parser = argparse.ArgumentParser(description="Responder preguntas de HotpotQA solo con el LLM")
    parser.add_argument(
        "--dataset",
        default="nlp-udesa/hotpot_qa_3k",
        help="Dataset de HuggingFace (default: nlp-udesa/hotpot_qa_3k)",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="Modelo LLM para litellm (usa OPENAI_API_KEY del entorno)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Limitar la cantidad de preguntas a responder (default: 10 para prueba)",
    )
    parser.add_argument(
        "--output",
        default="respuestas_llm.jsonl",
        help="Archivo de salida (default: respuestas_llm.jsonl)",
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

    respuestas = []

    for ejemplo in ds:
        pregunta = ejemplo["question"]

        # armar prompt y llamar al LLM
        prompt = armar_prompt(pregunta)
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
