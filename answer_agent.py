import argparse
import json
import os
import re
from datetime import datetime

from datasets import load_dataset
from dotenv import load_dotenv
from litellm import completion
import wikipedia

# carga OPENAI_API_KEY desde el archivo .env
load_dotenv()

# idioma inglés y user-agent requerido por la API de Wikipedia
wikipedia.set_lang("en")
wikipedia.set_user_agent("User-Agent: InfinitumBotty/0.1 (https://github.com/ctx77/InfinitumBotty) python-via-wikipedia-module/1.4.0")


# ── Tool 1: busca títulos de páginas en Wikipedia ─────────────────────────────
def wikipedia_search(query):
    resultados = wikipedia.search(query, results=5)   # devuelve lista de títulos
    return "Titles: " + ", ".join(resultados)


# ── Tool 2: devuelve el resumen de una página dado su título ──────────────────
def wikipedia_summary(title):
    try:
        return wikipedia.summary(title, sentences=3)  # primeras 3 oraciones
    except wikipedia.exceptions.DisambiguationError as e:
        return wikipedia.summary(e.options[0], sentences=3)  # si es ambiguo, toma la primera opción
    except Exception as e:
        return f"Error: {e}"


# ── Diccionario de tools disponibles para el agente ──────────────────────────
TOOLS = {
    "wikipedia_search": wikipedia_search,
    "wikipedia_summary": wikipedia_summary,
}


# ── System prompt: le dice al LLM cómo comportarse y qué formato usar ─────────
SYSTEM_PROMPT = """Sos un sistema de question answering para HotpotQA, un dataset de preguntas MULTI-HOP en inglés.

Tenés dos herramientas:
  wikipedia_search: busca páginas. Input: una query. Output: lista de títulos.
  wikipedia_summary: lee una página. Input: título exacto. Output: resumen de 3 oraciones.

Si la pregunta requiere dos búsquedas separadas (multi-hop), identificá los dos subtemas en el primer Thought y buscalos por separado.

Usá este formato en cada paso:
Thought: qué necesito saber
Action: wikipedia_search o wikipedia_summary
Action Input: el input de la tool
Observation: (lo agrega el sistema, no vos)

Cuando tengas la respuesta:
Thought: ya sé la respuesta
Final Answer: <respuesta>

IMPORTANTE — la línea "Final Answer:" debe contener ÚNICAMENTE la respuesta, sin razonamiento.

Reglas de formato:
- Respondé en inglés.
- Respondé solo con la respuesta, lo mas corta posible: una palabra, un nombre, una fecha, un numero o una frase breve.
- No escribas oraciones completas.
- No repitas la pregunta.
- No expliques ni justifiques.
- Para preguntas de si/no respondé exactamente 'yes' o 'no', en minuscula.
- Nunca digas 'I don't know' ni 'not enough information'. Siempre dá tu mejor respuesta.
- Identificá que tipo de dato pide la pregunta (un largo/distancia, una fecha, una cantidad, una persona, un lugar, un titulo) y devolvé exactamente ese tipo, no un dato relacionado.
- Para preguntas de longitud o distancia: respondé solo el número con unidades (ej: "65 mi", "30 km").
- Para preguntas de cuánto tiempo existió algo: si encontrás fechas de inicio y fin, respondé el rango (ej: "1874 until 1994"), no la duración calculada.
"""


# ── Loop ReACT: corre el ida y vuelta entre el LLM y las tools ────────────────
def react_loop(pregunta, model, max_steps=4):
    # armamos el historial de mensajes que se va a ir extendiendo en cada paso
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},   # instrucciones globales
        {"role": "user",   "content": f"Question: {pregunta}"},  # la pregunta
    ]

    for _ in range(max_steps):
        # llamamos al LLM; stop=["Observation:"] hace que se detenga antes de inventar la observación
        response = completion(model=model, messages=messages, temperature=0, stop=["Observation:"], max_tokens=300)
        text = response.choices[0].message.content.strip()  # texto generado por el LLM

        # agregamos la respuesta del LLM al historial
        messages.append({"role": "assistant", "content": text})

        # si el LLM escribió "Final Answer:", extraemos la respuesta y terminamos
        match_final = re.search(r"Final Answer:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
        if match_final:
            return match_final.group(1).strip()

        # si no terminó, parseamos qué tool quiere usar y con qué input
        match_action = re.search(r"Action:\s*(\w+)", text)
        match_input  = re.search(r"Action Input:\s*(.+?)(?:\n|$)", text)

        # ejecutamos la tool y guardamos el resultado como Observation
        tool_name   = match_action.group(1).strip()
        tool_input  = match_input.group(1).strip()
        observation = TOOLS[tool_name](tool_input)

        # agregamos la Observation al historial para que el LLM la vea en el próximo paso
        messages.append({"role": "user", "content": f"Observation: {observation}"})

    # si se agotaron los pasos, le pedimos la respuesta final directamente
    messages.append({"role": "user", "content": (
        "Dá la respuesta final ahora. Escribí SOLO 'Final Answer: <respuesta>', nada más."
    )})
    response = completion(model=model, messages=messages, temperature=0, max_tokens=100)
    text = response.choices[0].message.content.strip()
    match_final = re.search(r"Final Answer:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    if match_final:
        return match_final.group(1).strip()
    # último recurso: si el modelo no usó el formato, devolvemos la primera línea no vacía
    first_line = next((l.strip() for l in text.splitlines() if l.strip()), text.strip())
    return first_line


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",   default="nlp-udesa/hotpot_qa_3k")
    parser.add_argument("--model",     default="gpt-4o-mini")
    parser.add_argument("--max-steps", type=int, default=4)  # 4 pasos max para no explotar el contexto (límite fijo del servidor de OpenAI, no configurable)
    parser.add_argument("--limit",     type=int, default=None)
    parser.add_argument("--output",    default="answers_agente/respuestas_agente.jsonl")
    args = parser.parse_args()

    ### BEGIN SOLUTION

    # cargamos el split de validación del dataset
    ds = load_dataset(args.dataset)["validation"]
    if args.limit is not None:
        ds = ds.select(range(args.limit))  # tomamos solo las primeras N preguntas

    respuestas = []

    for i, ejemplo in enumerate(ds):
        pregunta = ejemplo["question"]
        print(f"[{i+1}/{len(ds)}] {pregunta}")

        # corremos el loop ReACT para esta pregunta
        respuesta_pred = react_loop(pregunta, model=args.model, max_steps=args.max_steps)
        print(f"  → pred: {respuesta_pred} | true: {ejemplo['answer']}\n")

        # guardamos el resultado con el formato que consume evaluar.py
        respuestas.append({
            "id":          ejemplo["id"],
            "question":    pregunta,
            "answer_pred": respuesta_pred,
            "answer_true": ejemplo["answer"],
        })

    # construimos la ruta de salida relativa a la ubicación del script
    output = args.output
    if not os.path.isabs(output):
        output = os.path.join(os.path.dirname(os.path.abspath(__file__)), output)

    # agregamos timestamp para no pisar corridas anteriores
    base, ext = os.path.splitext(output)
    output = f"{base}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{ext}"

    # escribimos el JSONL
    with open(output, "w") as f:
        for r in respuestas:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Listo. {len(respuestas)} respuestas guardadas en {output}")

    ### END SOLUTION


if __name__ == "__main__":
    main()
