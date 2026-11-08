"""
config.py
=========
Hiperparámetros y opciones de la GAN, reunidos en un solo lugar.

Todos se pueden cambiar desde la línea de comandos (ver __init__.py),
por ejemplo:  uv run gan-mnist --categoria 3 --epocas 50
"""

from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Nombres de las categorías (clases) de cada dataset.
# La GAN se entrena SOLO con las imágenes de UNA categoría, así el resultado
# es fácil de interpretar: "la GAN aprendió a dibujar el dígito 7".
# ---------------------------------------------------------------------------
CATEGORIAS = {
    "mnist": [f"dígito {i}" for i in range(10)],
    "fashion": [
        "Camiseta/top",   # 0
        "Pantalón",       # 1
        "Suéter",         # 2
        "Vestido",        # 3
        "Abrigo",         # 4
        "Sandalia",       # 5
        "Camisa",         # 6
        "Zapatilla",      # 7
        "Bolso",          # 8
        "Botín",          # 9
    ],
}


@dataclass
class Config:
    # --- Datos -------------------------------------------------------------
    dataset: str = "mnist"          # "mnist" (dígitos) o "fashion" (ropa)
    categoria: int = 7              # índice 0..9 de la categoría a aprender
    carpeta_datos: Path = Path("data")

    # --- Dimensiones -------------------------------------------------------
    dim_ruido: int = 100            # tamaño del vector latente z (entrada del Generador)
    alto: int = 28                  # MNIST: imágenes 28x28 en escala de grises
    ancho: int = 28

    # --- Entrenamiento -----------------------------------------------------
    epocas: int = 30
    tam_lote: int = 64              # batch size
    lr_generador: float = 2e-4      # tasa de aprendizaje del Generador
    lr_discriminador: float = 2e-4  # tasa de aprendizaje del Discriminador
    beta1: float = 0.5              # Adam beta1 (0.5 es lo habitual en GANs)
    beta2: float = 0.999            # Adam beta2
    semilla: int = 42               # para que los resultados sean reproducibles

    # --- Salidas -----------------------------------------------------------
    carpeta_resultados: Path = Path("resultados")
    guardar_cada: int = 5           # guarda una grilla de muestras cada N épocas
    num_muestras: int = 64          # nº de imágenes en la grilla final (8x8)

    @property
    def dim_imagen(self) -> int:
        """Cantidad de píxeles de una imagen aplanada: 28*28 = 784."""
        return self.alto * self.ancho

    @property
    def nombre_categoria(self) -> str:
        """Nombre legible de la categoría usada, p. ej. 'dígito 7'."""
        return CATEGORIAS[self.dataset][self.categoria]
