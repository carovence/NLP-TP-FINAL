import argparse                 
import json                    
import os                      
import re                       
import time                     
from datetime import datetime  

from datasets import load_dataset
from dotenv import load_dotenv
from litellm import completion
import wikipedia


load_dotenv()


wikipedia.set_lang("en")  
wikipedia.set_user_agent("User-Agent: InfinitumBotty/0.1 (https://github.com/ctx77/InfinitumBotty) python-via-wikipedia-module/1.4.0")  
wikipedia.set_rate_limiting(True)  


# ── Tool 1: busca títulos de páginas en Wikipedia ─────────────────────────────
def wikipedia_search(query, retries=3):
    for intento in range(retries):                          
        try:
            resultados = wikipedia.search(query, results=5)  
            return "Titles: " + ", ".join(resultados)       
        except Exception as e:                               
            if intento < retries - 1:                        
                time.sleep(2 * (intento + 1))               
                continue                                    
            return f"Error searching: {e}"                   


# ── Tool 2: devuelve el resumen de una página dado su título ──────────────────
def wikipedia_summary(title, retries=3):
    for intento in range(retries):                                          
        try:
            return wikipedia.summary(title, sentences=3, auto_suggest=False)  
        except wikipedia.exceptions.DisambiguationError as e:               
            try:
                return wikipedia.summary(e.options[0], sentences=3, auto_suggest=False)  
            except Exception:
                return f"Disambiguation: could not resolve '{title}'"       
        except wikipedia.exceptions.PageError as e:                         
            return f"Error: {e}"                                            
        except Exception as e:                                              
            if intento < retries - 1:                                       
                time.sleep(2 * (intento + 1))                               
                continue
            return f"Error: {e}"                                            



TOOLS = {
    "wikipedia_search": wikipedia_search,
    "wikipedia_summary": wikipedia_summary,
}



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



def react_loop(pregunta, model, max_steps=4):
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},   
        {"role": "user",   "content": f"Question: {pregunta}"},  
    ]

    for _ in range(max_steps):
        
        response = completion(model=model, messages=messages, temperature=0, stop=["Observation:"], max_tokens=300)
        text = response.choices[0].message.content.strip()  

        
        messages.append({"role": "assistant", "content": text})

        
        match_final = re.search(r"Final Answer:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
        if match_final:
            return match_final.group(1).strip()

        
        match_action = re.search(r"Action:\s*(\w+)", text)            
        match_input  = re.search(r"Action Input:\s*(.+?)(?:\n|$)", text)  

        
        if not match_action or not match_input:                       
            messages.append({"role": "user", "content":              
                "Formato inválido. Usá 'Action:' + 'Action Input:' o 'Final Answer:'."})
            continue                                                  
        tool_name  = match_action.group(1).strip()                    
        tool_input = match_input.group(1).strip()                     

        
        if tool_name not in TOOLS:                                    
            observation = f"Error: tool '{tool_name}' no existe. Usá: {', '.join(TOOLS)}"  
        else:
            observation = TOOLS[tool_name](tool_input)               

        
        messages.append({"role": "user", "content": f"Observation: {observation}"})

    
    messages.append({"role": "user", "content": (
        "Dá la respuesta final ahora. Escribí SOLO 'Final Answer: <respuesta>', nada más."
    )})
    response = completion(model=model, messages=messages, temperature=0, max_tokens=100)
    text = response.choices[0].message.content.strip()
    match_final = re.search(r"Final Answer:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    if match_final:
        return match_final.group(1).strip()
    
    first_line = next((l.strip() for l in text.splitlines() if l.strip()), text.strip())
    return first_line


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",   default="nlp-udesa/hotpot_qa_3k")
    parser.add_argument("--model",     default="gpt-4o-mini")
    parser.add_argument("--max-steps", type=int, default=4)  
    parser.add_argument("--start",     type=int, default=0)   
    parser.add_argument("--limit",     type=int, default=None) 
    parser.add_argument("--output",    default="answers_agente/respuestas_agente.jsonl")
    args = parser.parse_args()

    ### BEGIN SOLUTION

    
    ds = load_dataset(args.dataset)["validation"]   

    
    fin = args.limit if args.limit is not None else len(ds)  
    ds = ds.select(range(args.start, fin))          

    
    output = args.output                           
    if not os.path.isabs(output):                   
        output = os.path.join(os.path.dirname(os.path.abspath(__file__)), output)  

    base, ext = os.path.splitext(output)            
    output = f"{base}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{ext}"  
    os.makedirs(os.path.dirname(output), exist_ok=True)  
    total = 0                                       

    
    with open(output, "w") as f:                   
        for i, ejemplo in enumerate(ds):
            pregunta = ejemplo["question"]
            print(f"[{args.start + i + 1}/{fin}] {pregunta}")  

            
            respuesta_pred = react_loop(pregunta, model=args.model, max_steps=args.max_steps)
            print(f"  → pred: {respuesta_pred} | true: {ejemplo['answer']}\n")

            
            registro = {
                "id":          ejemplo["id"],
                "question":    pregunta,
                "answer_pred": respuesta_pred,
                "answer_true": ejemplo["answer"],
            }
            f.write(json.dumps(registro, ensure_ascii=False) + "\n")  
            f.flush()                               
            total += 1                             

            time.sleep(0.5)                        

    print(f"Listo. {total} respuestas guardadas en {output}")

    ### END SOLUTION


if __name__ == "__main__":
    main()
