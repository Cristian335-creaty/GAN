"""
inspeccion.py
=============
Herramientas para VER los parámetros y valores de cada red:

  * resumen_parametros()  -> tabla capa por capa: forma de los pesos,
                             nº de parámetros y estadísticas (media, desv.,
                             mín., máx.).  Se imprime y se guarda en .txt
  * valores_muestra()     -> valores concretos que producen las redes:
                             D(x) para imágenes reales, D(G(z)) para falsas
                             y rango de píxeles que genera G.
"""

from pathlib import Path

import torch
import torch.nn as nn


def resumen_parametros(modelo: nn.Module, nombre: str, archivo: Path | None = None) -> str:
    """Construye (e imprime) una tabla con todos los parámetros entrenables."""
    lineas = []
    sep = "-" * 104
    lineas.append(sep)
    lineas.append(f" PARÁMETROS DE LA RED: {nombre}")
    lineas.append(sep)
    lineas.append(
        f" {'Parámetro':<22}{'Forma':<16}{'Cantidad':>12}"
        f"{'Media':>12}{'Desv.':>12}{'Mín.':>12}{'Máx.':>12}"
    )
    lineas.append(sep)

    total = 0
    # named_parameters() recorre cada tensor entrenable (pesos y sesgos).
    #   "red.0.weight" = pesos W de la capa 0 del Sequential
    #   "red.0.bias"   = sesgos b de la capa 0
    #   En BatchNorm: weight = gamma, bias = beta
    for nombre_param, p in modelo.named_parameters():
        n = p.numel()
        total += n
        datos = p.detach().float()
        lineas.append(
            f" {nombre_param:<22}{str(tuple(p.shape)):<16}{n:>12,}"
            f"{datos.mean().item():>12.4f}{datos.std().item() if n > 1 else 0.0:>12.4f}"
            f"{datos.min().item():>12.4f}{datos.max().item():>12.4f}"
        )
    lineas.append(sep)
    lineas.append(f" TOTAL de parámetros entrenables: {total:,}")
    lineas.append(sep)

    # Estructura de capas tal como la ve PyTorch
    lineas.append(" Estructura:")
    lineas.extend("   " + l for l in str(modelo).splitlines())
    lineas.append("")

    texto = "\n".join(lineas)
    print(texto)
    if archivo is not None:
        archivo.parent.mkdir(parents=True, exist_ok=True)
        archivo.write_text(texto, encoding="utf-8")
    return texto


@torch.no_grad()
def valores_muestra(G: nn.Module, D: nn.Module, reales: torch.Tensor, z: torch.Tensor, titulo: str) -> dict:
    """Muestra valores concretos de las dos redes sobre unas pocas imágenes."""
    G.eval()
    D.eval()  # desactiva Dropout/BatchNorm de entrenamiento para medir

    falsas = G(z)
    d_real = D(reales).squeeze(1)
    d_falso = D(falsas).squeeze(1)

    print(f"\n===== VALORES DEL GENERADOR Y DEL DISCRIMINADOR ({titulo}) =====")
    print(" Generador  G(z):")
    print(f"   ruido z           -> forma {tuple(z.shape)}, media {z.mean():.3f}, desv. {z.std():.3f}")
    print(f"   imagen G(z)       -> forma {tuple(falsas.shape)}, "
          f"píxeles en [{falsas.min():.3f}, {falsas.max():.3f}], media {falsas.mean():.3f}")
    print(" Discriminador  D(.) = probabilidad de ser REAL:")
    print("   D(x)    reales :", " ".join(f"{v:.3f}" for v in d_real[:8].tolist()))
    print("   D(G(z)) falsas :", " ".join(f"{v:.3f}" for v in d_falso[:8].tolist()))
    print(f"   promedio D(x) = {d_real.mean():.3f}   |   promedio D(G(z)) = {d_falso.mean():.3f}")
    print("   (Ideal al converger: ambos cerca de 0.5 -> D ya no distingue)\n")

    G.train()
    D.train()
    return {"D_x": d_real.mean().item(), "D_G_z": d_falso.mean().item()}
