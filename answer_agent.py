import argparse
import json
import os

from datasets import load_dataset
from dotenv import load_dotenv
from litellm import completion
import wikipedia

# carga las variables del archivo .env (entre ellas OPENAI_API_KEY)
load_dotenv()


def armar_prompt(pregunta, contextos):
    # numeramos los pasajes de Wikipedia, de mas a menos relevante
    contexto = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contextos))
    return (
        "Sos un sistema de question answering para el dataset HotpotQA, "
        "un dataset de preguntas MULTI-HOP en inglés."
        "Te paso PASAJES recuperados de Wikipedia, ordenados de mas a menos relevante "
        "([1] es el mas relevante).\n\n"
        "Usá la información de los pasajes para responder; la respuesta puede requerir "
        "combinar varios pasajes. Si la respuesta no aparece explícita en los pasajes, "
        "inferí la mejor respuesta posible (no inventes datos imposibles).\n\n"
        "Reglas de formato:\n"
        "- Respondé en inglés.\n"
        "- Respondé solo con la respuesta, lo mas corta posible: una palabra, un nombre, "
        "una fecha, un numero o una frase breve.\n"
        "- No escribas oraciones completas.\n"
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
    parser = argparse.ArgumentParser(description="Responder preguntas de HotpotQA con un agente que consulta Wikipedia")
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
        "--n-results",
        type=int,
        default=3,
        help="Cantidad de paginas de Wikipedia a recuperar por pregunta (default: 3)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Limitar la cantidad de preguntas a responder (default: 10 para prueba)",
    )
    parser.add_argument(
        "--output",
        default="answers_agente/respuestas_agente.jsonl",
        help="Archivo de salida (default: answers_agente/respuestas_agente.jsonl)",
    )
    args = parser.parse_args()

    ### BEGIN SOLUTION

    # dataset
    ds = load_dataset(args.dataset)["validation"]
    if args.limit is not None:
        ds = ds.select(range(args.limit))

    # configuracion de Wikipedia (HotpotQA esta en ingles).
    # La API rechaza el User-Agent por defecto de la libreria y devuelve no-JSON;
    # con uno propio anda (si no, wikipedia.search tira JSONDecodeError).
    wikipedia.set_lang("en")
    wikipedia.set_user_agent("tp-nlp-udesa/1.0 (igarnica@udesa.edu.ar)")

    respuestas = []

    for ejemplo in ds:
        pregunta = ejemplo["question"]

        # recuperar pasajes de Wikipedia: top-k titulos + resumen de cada pagina.
        # todo en try/except para que una pagina ambigua / inexistente / un corte
        # de red no corte la corrida entera.
        contextos = []
        try:
            titulos = wikipedia.search(pregunta, results=args.n_results)
        except Exception:
            titulos = []
        for titulo in titulos:
            try:
                resumen = wikipedia.summary(titulo, sentences=5)
                contextos.append(f"{titulo}: {resumen}")
            except Exception:
                continue

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

    # anclamos la salida a la carpeta del script para que el JSON siempre
    # caiga en answers_agente/ del repo, sin importar desde donde se ejecute.
    output = args.output
    if not os.path.isabs(output):
        output = os.path.join(os.path.dirname(os.path.abspath(__file__)), output)

    # agregar timestamp al nombre para no pisar corridas anteriores
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base, ext = os.path.splitext(output)
    output = f"{base}_{timestamp}{ext}"

    # exportar al final
    with open(output, "w") as f:
        for r in respuestas:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

if __name__ == "__main__":
     main()      
