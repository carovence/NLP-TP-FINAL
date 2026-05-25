# %% [markdown]
# # TP Final - Multi-hop QA con RAG y Agentes
# ## Paso 0-1: Exploración del dataset HotPotQA

# %%
# Instalación de librerías
!pip install datasets -qqq

# %%
from datasets import load_dataset

# Cargamos HotPotQA en su configuración "distractor"
# (cada pregunta viene con 10 párrafos: 2 relevantes + 8 distractores)
ds = load_dataset("hotpotqa/hotpot_qa", "distractor")

print(ds)

# %% [markdown]
# ## Ver los splits y sus tamaños

# %%
# Vemos qué splits tiene y cuántos ejemplos hay en cada uno
for split in ds:
    print(f"{split}: {len(ds[split])} ejemplos")

# %% [markdown]
# ## Ver los campos de un ejemplo

# %%
# Miramos las columnas del split de validación
print("Campos:", ds["validation"].column_names)

# %%
# Miramos un ejemplo completo
ejemplo = ds["validation"][0]

print("PREGUNTA:")
print(ejemplo["question"])
print()
print("RESPUESTA:")
print(ejemplo["answer"])
print()
print("TIPO:", ejemplo["type"])
print("NIVEL:", ejemplo["level"])

# %% [markdown]
# ## Ver la estructura de los contextos

# %%
# El campo "context" tiene dos partes paralelas: title y sentences
contexto = ejemplo["context"]

print("Cantidad de párrafos:", len(contexto["title"]))
print()

# Recorremos cada párrafo (título + sus oraciones)
for i in range(len(contexto["title"])):
    titulo = contexto["title"][i]
    oraciones = contexto["sentences"][i]
    print(f"--- Párrafo {i}: {titulo} ---")
    print(" ".join(oraciones))
    print()

# %% [markdown]
# ## Ver los supporting facts (qué párrafos son los relevantes)

# %%
# supporting_facts indica qué oraciones se necesitan para responder
sf = ejemplo["supporting_facts"]

print("Títulos de soporte:", sf["title"])
print("Índices de oración:", sf["sent_id"])