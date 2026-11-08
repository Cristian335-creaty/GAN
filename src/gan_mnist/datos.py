"""
datos.py
========
Carga del dataset y filtrado por CATEGORÍA.

Se descarga MNIST (o Fashion-MNIST) con torchvision la primera vez y
se guarda en la carpeta ./data. Luego se conservan solo las imágenes
cuya etiqueta coincide con la categoría elegida.
"""

import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets

from .config import Config


def cargar_datos(cfg: Config) -> DataLoader:
    clase_dataset = datasets.MNIST if cfg.dataset == "mnist" else datasets.FashionMNIST
    dataset = clase_dataset(root=str(cfg.carpeta_datos), train=True, download=True)

    # ---- FILTRO POR CATEGORÍA -------------------------------------------
    # dataset.targets contiene la etiqueta (0..9) de cada imagen.
    # Nos quedamos solo con las de la categoría elegida.
    mascara = dataset.targets == cfg.categoria
    imagenes = dataset.data[mascara].float()        # [N, 28, 28], píxeles 0..255

    # ---- NORMALIZACIÓN A [-1, 1] ----------------------------------------
    # 0..255 -> 0..1 -> -1..1.  Se usa [-1, 1] porque la última capa del
    # Generador es Tanh, que también produce valores en [-1, 1]; así las
    # imágenes reales y las falsas quedan en la misma escala.
    imagenes = imagenes / 255.0
    imagenes = (imagenes - 0.5) / 0.5
    imagenes = imagenes.unsqueeze(1)                 # [N, 1, 28, 28] (1 canal)

    print(
        f"[Datos] Dataset: {cfg.dataset.upper()} | Categoría usada: "
        f"{cfg.categoria} = '{cfg.nombre_categoria}' | "
        f"Imágenes de entrenamiento: {len(imagenes)}"
    )

    # Solo necesitamos las imágenes (la etiqueta es siempre la misma).
    return DataLoader(
        TensorDataset(imagenes),
        batch_size=cfg.tam_lote,
        shuffle=True,
        drop_last=True,
    )


def ruido_latente(n: int, dim_ruido: int, dispositivo: torch.device) -> torch.Tensor:
    """Vector de ruido z ~ N(0, 1): la 'semilla' a partir de la cual el
    Generador crea una imagen. Forma: [n, dim_ruido]."""
    return torch.randn(n, dim_ruido, device=dispositivo)
