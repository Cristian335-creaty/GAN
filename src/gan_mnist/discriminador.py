"""
discriminador.py
================
>>> AQUÍ SE CREA LA RED NEURONAL DISCRIMINADORA (D) <<<

Función:  D(x) -> probabilidad de que x sea REAL
    Entrada : imagen de 1 x 28 x 28 (real o generada por G)
    Salida  : un número en [0, 1]
              ~1  => "creo que es real"
              ~0  => "creo que es falsa (la hizo el Generador)"

Es un clasificador binario: real vs. falso.

Arquitectura (perceptrón multicapa):

    imagen (1x28x28) -> se aplana a 784
      │  Linear 784 -> 512   + LeakyReLU(0.2) + Dropout(0.3)
      │  Linear 512 -> 256   + LeakyReLU(0.2) + Dropout(0.3)
      │  Linear 256 -> 1     + Sigmoid
      ▼
    probabilidad (1)
"""

import torch
import torch.nn as nn


class Discriminador(nn.Module):
    def __init__(self, dim_imagen: int = 784):
        super().__init__()

        # ------------------------------------------------------------------
        # CREACIÓN DE LA RED DISCRIMINADORA
        # ------------------------------------------------------------------
        self.red = nn.Sequential(
            # Capa oculta 1: recibe los 784 píxeles.
            # Parámetros: 784*512 + 512 = 401 920
            nn.Linear(dim_imagen, 512),
            nn.LeakyReLU(0.2, inplace=True),
            # Dropout apaga al azar el 30 % de las neuronas en cada paso;
            # evita que D memorice y se vuelva "demasiado fuerte" muy rápido.
            # (No tiene parámetros entrenables.)
            nn.Dropout(0.3),

            # Capa oculta 2: 512 -> 256.  Parámetros: 512*256 + 256 = 131 328
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            # Capa de salida: 256 -> 1.   Parámetros: 256*1 + 1 = 257
            nn.Linear(256, 1),
            # Sigmoid convierte el valor en una probabilidad entre 0 y 1.
            nn.Sigmoid(),
        )

    def forward(self, imagen: torch.Tensor) -> torch.Tensor:
        # [lote, 1, 28, 28] -> [lote, 784]
        plano = imagen.view(imagen.size(0), -1)
        # [lote, 784] -> [lote, 1]
        return self.red(plano)
