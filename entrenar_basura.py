import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm

from sklearn.metrics import classification_report, confusion_matrix


# -------------------------
# Configuración
# -------------------------
ruta_dataset = Path("dataset_split")  
tamano_lote = 32
epocas = 12
tasa_aprendizaje = 3e-4
semilla = 42
tamano_imagen = 224  # EfficientNet-B0 usa 224x224

usar_gpu = torch.cuda.is_available()
dispositivo = torch.device("cuda" if usar_gpu else "cpu")


# -------------------------
# Reproducibilidad
# -------------------------
def fijar_semillas(semilla: int):
    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(semilla)


fijar_semillas(semilla)


# -------------------------
# Transformaciones (augmentations)
# Se llevan las imagenes al tamaño correspondiente y se aumenta la cantidad de datos
# -------------------------
transformacion_entrenamiento = transforms.Compose([
    transforms.RandomResizedCrop(tamano_imagen, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.02),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

transformacion_eval = transforms.Compose([
    transforms.Resize((tamano_imagen, tamano_imagen)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# -------------------------
# Datasets y DataLoaders
# -------------------------
dataset_entrenamiento = datasets.ImageFolder(ruta_dataset / "train", transform=transformacion_entrenamiento)
dataset_validacion = datasets.ImageFolder(ruta_dataset / "val", transform=transformacion_eval)
dataset_prueba = datasets.ImageFolder(ruta_dataset / "test", transform=transformacion_eval)

clases = dataset_entrenamiento.classes
cantidad_clases = len(clases)
cargador_entrenamiento = DataLoader(dataset_entrenamiento, batch_size=tamano_lote, shuffle=True,  num_workers=0, pin_memory=False)
cargador_validacion   = DataLoader(dataset_validacion,   batch_size=tamano_lote, shuffle=False, num_workers=0, pin_memory=False)
cargador_prueba       = DataLoader(dataset_prueba,       batch_size=tamano_lote, shuffle=False, num_workers=0, pin_memory=False)

print("Clases:", clases)
print("Dispositivo:", dispositivo)


# -------------------------
# Pesos por clase, balancea clases con pocos datos como trash
# -------------------------
conteo_por_clase = np.bincount([etiqueta for _, etiqueta in dataset_entrenamiento.samples], minlength=cantidad_clases)
# Peso inverso: a menos muestras, más peso
pesos_por_clase = (conteo_por_clase.sum() / (conteo_por_clase + 1e-9))
pesos_por_clase = torch.tensor(pesos_por_clase, dtype=torch.float32).to(dispositivo)

print("Conteo por clase (train):", dict(zip(clases, conteo_por_clase.tolist())))


# -------------------------
# Modelo: EfficientNet-B0 preentrenada
# -------------------------
modelo = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
# Reemplazar la última capa
entrada_clasificador = modelo.classifier[1].in_features
modelo.classifier[1] = nn.Linear(entrada_clasificador, cantidad_clases)
modelo = modelo.to(dispositivo)


# -------------------------
# Actualiza los pesos del modelo para bajar el error. 
# -------------------------
funcion_perdida = nn.CrossEntropyLoss(weight=pesos_por_clase)
optimizador = torch.optim.AdamW(modelo.parameters(), lr=tasa_aprendizaje, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizador, mode="max", factor=0.5, patience=2)


# -------------------------
# Entrenamiento / Evaluación por época
# -------------------------
def evaluar(modelo, cargador):
    modelo.eval()
    perdidas = []
    aciertos = 0
    total = 0

    with torch.no_grad():
        for imagenes, etiquetas in cargador:
            imagenes = imagenes.to(dispositivo)
            etiquetas = etiquetas.to(dispositivo)

            salidas = modelo(imagenes)
            perdida = funcion_perdida(salidas, etiquetas)
            perdidas.append(perdida.item())

            predicciones = salidas.argmax(dim=1)
            aciertos += (predicciones == etiquetas).sum().item()
            total += etiquetas.size(0)

    perdida_media = float(np.mean(perdidas)) if perdidas else 0.0
    exactitud = aciertos / max(total, 1)
    return perdida_media, exactitud


mejor_exactitud_val = 0.0
ruta_modelo_mejor = Path("modelo_mejor.pth")

for epoca in range(1, epocas + 1):
    modelo.train()
    perdidas_epoca = []
    barra = tqdm(cargador_entrenamiento, desc=f"Época {epoca}/{epocas}", leave=False)

    for imagenes, etiquetas in barra:
        imagenes = imagenes.to(dispositivo)
        etiquetas = etiquetas.to(dispositivo)

        optimizador.zero_grad()
        salidas = modelo(imagenes)
        perdida = funcion_perdida(salidas, etiquetas)
        perdida.backward()
        optimizador.step()

        perdidas_epoca.append(perdida.item())
        barra.set_postfix(perdida=float(np.mean(perdidas_epoca)))

    perdida_train = float(np.mean(perdidas_epoca)) if perdidas_epoca else 0.0
    perdida_val, exactitud_val = evaluar(modelo, cargador_validacion)

    scheduler.step(exactitud_val)

    print(f"[Época {epoca}] train_loss={perdida_train:.4f} | val_loss={perdida_val:.4f} | val_acc={exactitud_val:.4f}")

    # Guardar el mejor modelo según validación
    if exactitud_val > mejor_exactitud_val:
        mejor_exactitud_val = exactitud_val
        torch.save({
            "estado_modelo": modelo.state_dict(),
            "clases": clases,
            "tamano_imagen": tamano_imagen
        }, ruta_modelo_mejor)
        print(f"  -> Guardado mejor modelo en {ruta_modelo_mejor} (val_acc={mejor_exactitud_val:.4f})")


# -------------------------
# Evaluación final en TEST (con métricas)
# -------------------------
print("\nCargando mejor modelo para evaluar en TEST...")
checkpoint = torch.load(ruta_modelo_mejor, map_location=dispositivo)
modelo.load_state_dict(checkpoint["estado_modelo"])
modelo.eval()

etiquetas_reales = []
etiquetas_predichas = []

with torch.no_grad():
    for imagenes, etiquetas in cargador_prueba:
        imagenes = imagenes.to(dispositivo)
        salidas = modelo(imagenes)
        predicciones = salidas.argmax(dim=1).cpu().numpy()

        etiquetas_predichas.extend(predicciones.tolist())
        etiquetas_reales.extend(etiquetas.numpy().tolist())

print("\nReporte de clasificación (TEST):")
print(classification_report(etiquetas_reales, etiquetas_predichas, target_names=clases, digits=4))

matriz = confusion_matrix(etiquetas_reales, etiquetas_predichas)
print("Matriz de confusión (filas=real, columnas=predicho):")
print(matriz)
