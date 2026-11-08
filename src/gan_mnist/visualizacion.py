"""
visualizacion.py
================
Gráficas que se guardan en la carpeta ./resultados:

  * perdidas.png             -> curvas de pérdida de G y D
  * valores_D.png            -> evolución de D(x) y D(G(z))
  * histograma_pesos.png     -> distribución de los pesos de cada red
  * muestras_epoca_XXX.png   -> imágenes generadas durante el entrenamiento
  * resultado_final.png      -> resultado de la GAN con la categoría usada
"""

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # dibuja a archivo, sin abrir ventanas
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


def guardar_grilla(imagenes: torch.Tensor, ruta: Path, titulo: str) -> None:
    """Guarda una grilla cuadrada de imágenes (valores en [-1, 1])."""
    imagenes = imagenes.detach().cpu()
    imagenes = (imagenes + 1) / 2  # [-1, 1] -> [0, 1] para mostrarlas
    n = imagenes.size(0)
    lado = int(math.ceil(math.sqrt(n)))

    fig, ejes = plt.subplots(lado, lado, figsize=(lado * 1.1, lado * 1.1 + 0.6))
    for i, eje in enumerate(ejes.flat):
        eje.axis("off")
        if i < n:
            eje.imshow(imagenes[i, 0], cmap="gray", vmin=0, vmax=1)
    fig.suptitle(titulo, fontsize=11)
    fig.tight_layout()
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=110)
    plt.close(fig)


def comparar_reales_falsas(reales: torch.Tensor, falsas: torch.Tensor, ruta: Path, titulo: str) -> None:
    """Fila superior: imágenes REALES de la categoría.  Fila inferior: GENERADAS."""
    reales = (reales.detach().cpu() + 1) / 2
    falsas = (falsas.detach().cpu() + 1) / 2
    n = min(8, reales.size(0), falsas.size(0))
    fig, ejes = plt.subplots(2, n, figsize=(n * 1.3, 3.4))
    for i in range(n):
        ejes[0, i].imshow(reales[i, 0], cmap="gray", vmin=0, vmax=1)
        ejes[1, i].imshow(falsas[i, 0], cmap="gray", vmin=0, vmax=1)
        ejes[0, i].axis("off")
        ejes[1, i].axis("off")
    ejes[0, 0].set_title("Reales", loc="left", fontsize=9)
    ejes[1, 0].set_title("Generadas por G", loc="left", fontsize=9)
    fig.suptitle(titulo, fontsize=11)
    fig.tight_layout()
    fig.savefig(ruta, dpi=110)
    plt.close(fig)


def graficar_perdidas(historial: dict, ruta: Path, categoria: str) -> None:
    epocas = range(1, len(historial["perdida_D"]) + 1)
    fig, eje = plt.subplots(figsize=(8, 4.5))
    eje.plot(epocas, historial["perdida_D"], label="Pérdida Discriminador  L_D", color="#c0392b")
    eje.plot(epocas, historial["perdida_G"], label="Pérdida Generador  L_G", color="#2471a3")
    eje.plot(epocas, historial["perdida_D_real"], "--", label="L_D parte real  -log D(x)", color="#e59866")
    eje.plot(epocas, historial["perdida_D_falsa"], ":", label="L_D parte falsa  -log(1-D(G(z)))", color="#922b21")
    # Líneas de referencia del equilibrio teórico (D = 0.5)
    eje.axhline(math.log(2), color="gray", lw=0.8, ls="--")
    eje.text(1, math.log(2) + 0.03, "ln 2 ≈ 0.693 (equilibrio de L_G)", fontsize=8, color="gray")
    eje.axhline(2 * math.log(2), color="gray", lw=0.8, ls="--")
    eje.text(1, 2 * math.log(2) + 0.03, "2·ln 2 ≈ 1.386 (equilibrio de L_D)", fontsize=8, color="gray")
    eje.set_xlabel("Época")
    eje.set_ylabel("Pérdida (BCE)")
    eje.set_title(f"Funciones de pérdida de la GAN — categoría: {categoria}")
    eje.legend(fontsize=8)
    eje.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ruta, dpi=110)
    plt.close(fig)


def graficar_valores_D(historial: dict, ruta: Path, categoria: str) -> None:
    epocas = range(1, len(historial["D_x"]) + 1)
    fig, eje = plt.subplots(figsize=(8, 4))
    eje.plot(epocas, historial["D_x"], label="D(x): prob. asignada a REALES", color="#1e8449")
    eje.plot(epocas, historial["D_G_z"], label="D(G(z)): prob. asignada a FALSAS", color="#7d3c98")
    eje.axhline(0.5, color="gray", ls="--", lw=0.8, label="0.5 = D no distingue")
    eje.set_ylim(0, 1)
    eje.set_xlabel("Época")
    eje.set_ylabel("Salida del Discriminador")
    eje.set_title(f"Valores del Discriminador — categoría: {categoria}")
    eje.legend(fontsize=8)
    eje.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ruta, dpi=110)
    plt.close(fig)


def histograma_pesos(G: nn.Module, D: nn.Module, ruta: Path) -> None:
    """Distribución de todos los PESOS (W) de las capas Linear de cada red."""
    def pesos(modelo):
        return torch.cat([
            m.weight.detach().flatten().cpu() for m in modelo.modules() if isinstance(m, nn.Linear)
        ]).numpy()

    fig, ejes = plt.subplots(1, 2, figsize=(10, 3.8))
    ejes[0].hist(pesos(G), bins=120, color="#2471a3")
    ejes[0].set_title("Pesos del Generador")
    ejes[1].hist(pesos(D), bins=120, color="#c0392b")
    ejes[1].set_title("Pesos del Discriminador")
    for e in ejes:
        e.set_xlabel("valor del peso")
        e.set_ylabel("frecuencia")
        e.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ruta, dpi=110)
    plt.close(fig)
