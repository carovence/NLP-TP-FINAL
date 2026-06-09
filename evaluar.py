import argparse
import json
import re
import string
from collections import Counter


#normalizamos el texto

#pasamos a minusculas
def a_minusculas(texto):
    return texto.lower()

def sacar_puntuacion(texto):
    signos = set(string.punctuation)
    resultado = ""
    for char in texto:
        if char not in signos:
            resultado += char
    return resultado

#sacamos articulos
def sacar_articulos(texto):
    return re.sub(r"\b(a|an|the)\b", " ", texto)

#sacamos los espacios
def normalizar_espacios(texto):
    palabras = texto.split()
    return " ".join(palabras)

def normalizar(texto):
    texto = a_minusculas(texto)
    texto = sacar_puntuacion(texto)
    texto = sacar_articulos(texto)
    texto = normalizar_espacios(texto)
    return texto

# EM y F1

#vemos si son exactamente iguales las rtas
def exact_match(pred, true):
    if normalizar(pred) == normalizar(true):
        return 1
    return 0

#cuantos tokens en comun tienen las respuestas
def f1_score(pred, true):
    pred_tokens = normalizar(pred).split()
    true_tokens = normalizar(true).split()

    # Cuento tokens en comun (con multiplicidad).
    # Counter & Counter te devuelve el minimo por cada token.
    comunes = Counter(pred_tokens) & Counter(true_tokens)
    n_comunes = sum(comunes.values())

    if n_comunes == 0:
        return 0.0

    precision = n_comunes / len(pred_tokens)
    recall = n_comunes / len(true_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1


#leemos los json que tienen los campos answer_pred y answer_true y calculamos metricas 
def evaluar_archivo(path, mostrar_detalle=False):
    em_total = 0.0
    f1_total = 0.0
    n_preguntas = 0

    with open(path) as f:
        for linea in f:
            ejemplo = json.loads(linea)
            pred = ejemplo["answer_pred"]
            true = ejemplo["answer_true"]

            em = exact_match(pred, true)
            f1 = f1_score(pred, true)

            em_total += em
            f1_total += f1
            n_preguntas += 1

            if mostrar_detalle:
                marca = "OK" if em == 1 else "  "
                print(f"  [{marca}] EM={em} F1={f1:.2f} | pred={pred!r} | true={true!r}")

    em_promedio = 100.0 * em_total / n_preguntas
    f1_promedio = 100.0 * f1_total / n_preguntas

    return {
        "archivo": path,
        "n": n_preguntas,
        "EM": em_promedio,
        "F1": f1_promedio,
    }

# main

def main():
    parser = argparse.ArgumentParser(
        description="Evaluación de EM y F1 sobre las salidas de los distintos modelos."
    )
    parser.add_argument(
        "archivos",
        nargs="+",
        help="Uno o mas archivos .jsonl con campos answer_pred y answer_true"
    )
    parser.add_argument(
        "--detalle",
        action="store_true",
        help="Mostrar pregunta por pregunta (no solo el resumen)"
    )
    args = parser.parse_args()

    resultados = []
    for path in args.archivos:
        if args.detalle:
            print(f"\n=== {path} ===")
        res = evaluar_archivo(path, mostrar_detalle=args.detalle)
        resultados.append(res)

    # tabla resumen con metricas
    print("\n" + "=" * 60)
    print(f"{'archivo':<35} {'n':>4} {'EM':>8} {'F1':>8}")
    print("-" * 60)
    for r in resultados:
        print(f"{r['archivo']:<35} {r['n']:>4} {r['EM']:>7.1f}% {r['F1']:>7.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()