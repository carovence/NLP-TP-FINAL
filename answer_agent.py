import argparse                 # parsea los argumentos de linea de comandos (--limit, --model, etc.)
import json                     # serializa cada respuesta a JSON para el .jsonl
import os                       # manejo de rutas (anclar el output a la carpeta del script)
import re                       # expresiones regulares para parsear la salida del LLM
import time                     # sleep() para los reintentos con backoff y el rate limiting
from datetime import datetime   # timestamp en el nombre del archivo de salida

from datasets import load_dataset
from dotenv import load_dotenv
from litellm import completion
import wikipedia

# carga OPENAI_API_KEY desde el archivo .env
load_dotenv()

# idioma inglés y user-agent requerido por la API de Wikipedia
wikipedia.set_lang("en")  # todas las consultas se hacen contra en.wikipedia.org
wikipedia.set_user_agent("User-Agent: InfinitumBotty/0.1 (https://github.com/ctx77/InfinitumBotty) python-via-wikipedia-module/1.4.0")  # la API rechaza el UA por defecto y devuelve no-JSON
wikipedia.set_rate_limiting(True)  # FIX #4: espacia las requests para evitar que la API corte la conexion (RemoteDisconnected)


# ── Tool 1: busca títulos de páginas en Wikipedia ─────────────────────────────
def wikipedia_search(query, retries=3):
    for intento in range(retries):                          # FIX #1: hasta 3 intentos ante cortes de red
        try:
            resultados = wikipedia.search(query, results=5)  # devuelve una lista de titulos
            return "Titles: " + ", ".join(resultados)        # los unimos en un string para el LLM
        except Exception as e:                               # captura RemoteDisconnected, ConnectionError, etc.
            if intento < retries - 1:                        # si quedan intentos, esperamos y reintentamos
                time.sleep(2 * (intento + 1))                # backoff incremental: 2s, 4s
                continue                                     # vuelve al for para reintentar
            return f"Error searching: {e}"                   # agotados los intentos: devolvemos el error como Observation (NO rompe el loop)


# ── Tool 2: devuelve el resumen de una página dado su título ──────────────────
def wikipedia_summary(title, retries=3):
    for intento in range(retries):                                          # FIX #1: mismos reintentos que la otra tool
        try:
            return wikipedia.summary(title, sentences=3, auto_suggest=False)  # resumen de 3 oraciones del titulo exacto
        except wikipedia.exceptions.DisambiguationError as e:               # el titulo es ambiguo (varias paginas posibles)
            try:
                return wikipedia.summary(e.options[0], sentences=3, auto_suggest=False)  # probamos la primera opcion sugerida
            except Exception:
                return f"Disambiguation: could not resolve '{title}'"       # ni asi: devolvemos aviso (no rompe)
        except wikipedia.exceptions.PageError as e:                         # el titulo no existe: no tiene sentido reintentar
            return f"Error: {e}"                                            # devolvemos el error directo
        except Exception as e:                                              # error de red u otro transitorio
            if intento < retries - 1:                                       # si quedan intentos, backoff y reintento
                time.sleep(2 * (intento + 1))                               # 2s, 4s
                continue
            return f"Error: {e}"                                            # agotados los intentos: error como Observation


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
        match_action = re.search(r"Action:\s*(\w+)", text)            # capturamos el nombre de la tool
        match_input  = re.search(r"Action Input:\s*(.+?)(?:\n|$)", text)  # capturamos el input de la tool

        # FIX #2: si el LLM no respetó el formato, match_* es None y .group(1) rompería con AttributeError
        if not match_action or not match_input:                       # falta Action o Action Input
            messages.append({"role": "user", "content":              # le pedimos que reintente con el formato correcto
                "Formato inválido. Usá 'Action:' + 'Action Input:' o 'Final Answer:'."})
            continue                                                  # saltamos al proximo paso del loop sin ejecutar tool

        tool_name  = match_action.group(1).strip()                    # nombre de la tool (ya sabemos que existe el match)
        tool_input = match_input.group(1).strip()                     # input de la tool

        # FIX #2: si el LLM inventa un nombre de tool, TOOLS[tool_name] tiraría KeyError
        if tool_name not in TOOLS:                                    # tool inexistente
            observation = f"Error: tool '{tool_name}' no existe. Usá: {', '.join(TOOLS)}"  # avisamos las validas
        else:
            observation = TOOLS[tool_name](tool_input)                # ejecutamos la tool elegida

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
    parser.add_argument("--start",     type=int, default=0)   # indice (0-based) desde donde arrancar: para RETOMAR una corrida cortada
    parser.add_argument("--limit",     type=int, default=None) # indice EXCLUSIVO donde terminar (None = hasta el final del split)
    parser.add_argument("--output",    default="answers_agente/respuestas_agente.jsonl")
    args = parser.parse_args()

    ### BEGIN SOLUTION

    # cargamos el split de validación del dataset
    ds = load_dataset(args.dataset)["validation"]   # HotpotQA 3k, split de validacion

    # recortamos el rango [start, fin): start permite RETOMAR desde donde se cortó la corrida
    fin = args.limit if args.limit is not None else len(ds)  # fin exclusivo (default = todo el split)
    ds = ds.select(range(args.start, fin))          # ej: --start 528 arranca en la pregunta 528 (0-based)

    # FIX #3: construimos la ruta de salida ANTES del loop para poder ir escribiendo a medida que avanzamos
    output = args.output                            # ruta pedida por el usuario (relativa o absoluta)
    if not os.path.isabs(output):                   # si es relativa...
        output = os.path.join(os.path.dirname(os.path.abspath(__file__)), output)  # ...la anclamos a la carpeta del script

    base, ext = os.path.splitext(output)            # separamos nombre y extension
    output = f"{base}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{ext}"  # timestamp para no pisar corridas previas
    os.makedirs(os.path.dirname(output), exist_ok=True)  # FIX #3: aseguramos que la carpeta de salida exista

    total = 0                                       # contador de respuestas escritas

    # FIX #3: abrimos el archivo UNA vez y escribimos cada respuesta apenas la tenemos.
    # Asi, si la corrida se corta (red, etc.), conservamos todo el progreso hasta ese punto.
    with open(output, "w") as f:                    # modo escritura; el flush por linea preserva el progreso
        for i, ejemplo in enumerate(ds):
            pregunta = ejemplo["question"]
            print(f"[{args.start + i + 1}/{fin}] {pregunta}")  # numeracion absoluta del split (no relativa al recorte)

            # corremos el loop ReACT para esta pregunta
            respuesta_pred = react_loop(pregunta, model=args.model, max_steps=args.max_steps)
            print(f"  → pred: {respuesta_pred} | true: {ejemplo['answer']}\n")

            # armamos el registro con el formato exacto que consume evaluar.py
            registro = {
                "id":          ejemplo["id"],
                "question":    pregunta,
                "answer_pred": respuesta_pred,
                "answer_true": ejemplo["answer"],
            }
            f.write(json.dumps(registro, ensure_ascii=False) + "\n")  # escribimos la linea JSON
            f.flush()                               # FIX #3: forzamos el volcado a disco en cada paso
            total += 1                              # sumamos al contador

            time.sleep(0.5)                         # FIX #4: pausa entre preguntas para no saturar la API de Wikipedia

    print(f"Listo. {total} respuestas guardadas en {output}")

    ### END SOLUTION


if __name__ == "__main__":
    main()
