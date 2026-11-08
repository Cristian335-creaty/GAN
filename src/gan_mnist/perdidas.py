"""
perdidas.py
===========
>>> FUNCIONES DE PÉRDIDA DE LA GAN <<<

Ambas redes usan la Entropía Cruzada Binaria (BCE, Binary Cross-Entropy):

    BCE(p, y) = -[ y * log(p) + (1 - y) * log(1 - p) ]

    p = salida del Discriminador (probabilidad de "real")
    y = etiqueta objetivo (1 = real, 0 = falso)

La GAN es un juego minimax entre dos jugadores:

    min_G  max_D   E[ log D(x) ] + E[ log(1 - D(G(z))) ]

---------------------------------------------------------------------------
1) Pérdida del DISCRIMINADOR  (quiere acertar: reales->1, falsas->0)

    L_D = BCE(D(x), 1)        + BCE(D(G(z)), 0)
        = -log D(x)           - log(1 - D(G(z)))
          └ parte REAL ┘        └ parte FALSA ┘

    * Si D acierta en todo: D(x)≈1, D(G(z))≈0  ->  L_D ≈ 0
    * Si D no distingue nada: D(.)=0.5         ->  L_D = 2·ln 2 ≈ 1.386

2) Pérdida del GENERADOR  (quiere engañar: que D diga 1 a sus falsas)

    L_G = BCE(D(G(z)), 1) = -log D(G(z))

    Es la versión "no saturante" propuesta por Goodfellow et al. (2014):
    en lugar de minimizar log(1 - D(G(z))) (cuyo gradiente casi desaparece
    al inicio, cuando D rechaza fácilmente las falsas), se maximiza
    log D(G(z)), que da gradientes fuertes desde el principio.

    * Si G engaña a D: D(G(z))≈1  ->  L_G ≈ 0
    * Equilibrio teórico D(.)=0.5 ->  L_G = ln 2 ≈ 0.693
---------------------------------------------------------------------------
"""

import torch
import torch.nn as nn

# nn.BCELoss calcula el promedio de la BCE sobre el lote.
# (Se usa BCELoss porque el Discriminador ya termina en Sigmoid.)
bce = nn.BCELoss()


def perdida_discriminador(d_real: torch.Tensor, d_falso: torch.Tensor):
    """
    d_real  = D(x)     -> salida de D para imágenes REALES
    d_falso = D(G(z))  -> salida de D para imágenes FALSAS
    Devuelve (pérdida total, parte real, parte falsa) para poder graficarlas.
    """
    unos = torch.ones_like(d_real)     # etiqueta "real"
    ceros = torch.zeros_like(d_falso)  # etiqueta "falso"

    perdida_real = bce(d_real, unos)     # -log D(x)
    perdida_falsa = bce(d_falso, ceros)  # -log(1 - D(G(z)))
    return perdida_real + perdida_falsa, perdida_real, perdida_falsa


def perdida_generador(d_falso: torch.Tensor) -> torch.Tensor:
    """
    d_falso = D(G(z)).  El Generador "finge" que sus imágenes son reales
    (etiqueta 1) y mide qué tan lejos está D de creérselo.
    """
    unos = torch.ones_like(d_falso)
    return bce(d_falso, unos)            # -log D(G(z))
