import argparse
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


def construir_modelo(cantidad_clases: int):
    modelo = models.efficientnet_b0(weights=None)
    entrada_clasificador = modelo.classifier[1].in_features
    modelo.classifier[1] = nn.Linear(entrada_clasificador, cantidad_clases)
    return modelo


def cargar_checkpoint(ruta_checkpoint: Path, dispositivo: torch.device):
    checkpoint = torch.load(ruta_checkpoint, map_location=dispositivo)

    clases = checkpoint.get("clases", None)
    tamano_imagen = checkpoint.get("tamano_imagen", 224)

    if clases is None:
        raise ValueError("El checkpoint no contiene 'clases'. Re-entrená guardando clases o agregalas al checkpoint.")

    modelo = construir_modelo(len(clases))
    modelo.load_state_dict(checkpoint["estado_modelo"])
    modelo.to(dispositivo)
    modelo.eval()

    return modelo, clases, tamano_imagen


def preparar_transformacion(tamano_imagen: int):
    return transforms.Compose([
        transforms.Resize((tamano_imagen, tamano_imagen)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def clasificar_imagen(modelo, clases, ruta_imagen: Path, transformacion, dispositivo, topk: int = 3):
    imagen = Image.open(ruta_imagen).convert("RGB")
    tensor = transformacion(imagen).unsqueeze(0).to(dispositivo)  # (1, C, H, W)

    with torch.no_grad():
        logits = modelo(tensor)
        probabilidades = torch.softmax(logits, dim=1).squeeze(0)  # (num_clases,)

    topk = min(topk, len(clases))
    valores, indices = torch.topk(probabilidades, k=topk)

    resultados = []
    for valor, idx in zip(valores.cpu().tolist(), indices.cpu().tolist()):
        resultados.append((clases[idx], float(valor)))

    clase_predicha = resultados[0][0]
    confianza = resultados[0][1]
    return clase_predicha, confianza, resultados


def main():
    parser = argparse.ArgumentParser(description="Clasificar una imagen de basura con un modelo entrenado.")
    parser.add_argument("--modelo", type=str, default="modelo_mejor.pth", help="Ruta al checkpoint .pth")
    parser.add_argument("--imagen", type=str, required=True, help="Ruta a la imagen a clasificar")
    parser.add_argument("--topk", type=int, default=3, help="Cantidad de clases a mostrar (top-k)")
    args = parser.parse_args()

    ruta_modelo = Path(args.modelo)
    ruta_imagen = Path(args.imagen)

    if not ruta_modelo.exists():
        raise FileNotFoundError(f"No existe el modelo: {ruta_modelo}")
    if not ruta_imagen.exists():
        raise FileNotFoundError(f"No existe la imagen: {ruta_imagen}")

    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    modelo, clases, tamano_imagen = cargar_checkpoint(ruta_modelo, dispositivo)
    transformacion = preparar_transformacion(tamano_imagen)

    clase, confianza, top = clasificar_imagen(
        modelo=modelo,
        clases=clases,
        ruta_imagen=ruta_imagen,
        transformacion=transformacion,
        dispositivo=dispositivo,
        topk=args.topk
    )

    print(f"\nImagen: {ruta_imagen}")
    print(f"Predicción: {clase}  (confianza: {confianza:.4f})")
    print("\nTop resultados:")
    for nombre, prob in top:
        print(f" - {nombre:10s}  prob={prob:.4f}")


if __name__ == "__main__":
    main()
