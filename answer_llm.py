import argparse
import json
import os
import time

from datasets import load_dataset
from dotenv import load_dotenv
from litellm import completion

# carga las variables del archivo .env (entre ellas OPENAI_API_KEY)
load_dotenv()


def llamar_llm(model, prompt, intentos=4):
    """Llama al LLM con reintentos ante errores transitorios (rate limit, timeout, red).
    Espera incremental (1s, 2s, 4s, ...). Si fallan todos los intentos, devuelve None."""
    for i in range(intentos):
        try:
            response = completion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            return response.choices[0].message.content
        except Exception as e:
            if i == intentos - 1:
                print(f"  ⚠️ fallaron los {intentos} intentos: {e}")
                return None
            espera = 2 ** i
            print(f"  ⚠️ error en llamada LLM ({e}); reintento {i + 1}/{intentos} en {espera}s")
            time.sleep(espera)


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
        default=3000,
        help="Limitar la cantidad de preguntas a responder (default: 10 para prueba)",
    )
    parser.add_argument(
        "--output-dir",
        default="answers_llm",
        help="Directorio de salida (default: answers_llm)",
    )
    parser.add_argument(
        "--output",
        default="respuestas_llm.jsonl",
        help="Nombre base del archivo; se le agrega un sufijo con la cantidad (ej: respuestas_llm_50.jsonl)",
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

    # armar el path de salida: answers_llm/respuestas_llm_<cantidad>.jsonl
    os.makedirs(args.output_dir, exist_ok=True)
    base, ext = os.path.splitext(args.output)
    output_path = os.path.join(args.output_dir, f"{base}_{len(ds)}{ext}")

    # resume: si el archivo ya existe, no repetimos las preguntas ya respondidas
    ids_hechos = set()
    if os.path.exists(output_path):
        with open(output_path) as f:
            for linea in f:
                try:
                    ids_hechos.add(json.loads(linea)["id"])
                except (json.JSONDecodeError, KeyError):
                    pass
        print(f"Reanudando: ya hay {len(ids_hechos)} respuestas en {output_path}")

    n_nuevas = 0

    # abrimos en modo append y escribimos cada respuesta apenas la tenemos (incremental):
    # si se corta la corrida, lo ya hecho queda en disco y se puede retomar.
    with open(output_path, "a") as f_out:
        for ejemplo in ds:
            if ejemplo["id"] in ids_hechos:
                continue  # ya respondida en una corrida anterior

            pregunta = ejemplo["question"]

            # armar prompt y llamar al LLM (con reintentos)
            prompt = armar_prompt(pregunta)
            respuesta_pred = llamar_llm(args.model, prompt)
            if respuesta_pred is None:
                respuesta_pred = ""  # no frenamos toda la corrida por un caso fallido

            f_out.write(json.dumps({
                "id": ejemplo["id"],
                "question": pregunta,
                "answer_pred": respuesta_pred,
                "answer_true": ejemplo["answer"],
            }, ensure_ascii=False) + "\n")
            f_out.flush()  # forzamos el guardado en disco en cada paso
            n_nuevas += 1

    total = len(ids_hechos) + n_nuevas
    print(f"Listo. {n_nuevas} respuestas nuevas ({total} en total) en {output_path}.")

    ### END SOLUTION


if __name__ == "__main__":
    main()
