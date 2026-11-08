"""
generador.py
============
>>> AQUÍ SE CREA LA RED NEURONAL GENERADORA (G) <<<

Función:  G(z) -> imagen falsa
    Entrada : vector de ruido z de tamaño 100  (z ~ Normal(0, 1))
    Salida  : imagen de 1 x 28 x 28 con valores en [-1, 1]

Objetivo: producir imágenes tan parecidas a las reales que el
Discriminador no pueda distinguirlas.

Arquitectura (perceptrón multicapa, totalmente conectado):

    z (100)
      │  Linear 100 -> 256      + LeakyReLU(0.2)
      │  Linear 256 -> 512      + BatchNorm + LeakyReLU(0.2)
      │  Linear 512 -> 1024     + BatchNorm + LeakyReLU(0.2)
      │  Linear 1024 -> 784     + Tanh
      ▼
    imagen (784 = 28x28)  ->  se reacomoda a [1, 28, 28]

Parámetros entrenables de cada capa Linear:
    pesos  W : (salidas x entradas)
    sesgos b : (salidas)
    total    = entradas*salidas + salidas
Parámetros de cada BatchNorm1d: gamma (escala) y beta (desplazamiento),
uno por neurona -> 2 * neuronas.
"""

import torch
import torch.nn as nn


class Generador(nn.Module):
    def __init__(self, dim_ruido: int = 100, dim_imagen: int = 784, forma=(1, 28, 28)):
        super().__init__()
        self.forma = forma

        # ------------------------------------------------------------------
        # CREACIÓN DE LA RED GENERADORA
        # nn.Sequential apila las capas en el orden en que se aplican.
        # ------------------------------------------------------------------
        self.red = nn.Sequential(
            # Capa oculta 1: expande el ruido de 100 a 256 neuronas.
            # Parámetros: 100*256 + 256 = 25 856
            nn.Linear(dim_ruido, 256),
            # LeakyReLU deja pasar un poco de gradiente en valores negativos
            # (pendiente 0.2) y evita "neuronas muertas".
            nn.LeakyReLU(0.2, inplace=True),

            # Capa oculta 2: 256 -> 512.   Parámetros: 256*512 + 512 = 131 584
            nn.Linear(256, 512),
            # BatchNorm estabiliza el entrenamiento normalizando cada neurona.
            # Parámetros: 512 (gamma) + 512 (beta) = 1 024
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # Capa oculta 3: 512 -> 1024.  Parámetros: 512*1024 + 1024 = 525 312
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),          # Parámetros: 2 048
            nn.LeakyReLU(0.2, inplace=True),

            # Capa de salida: 1024 -> 784 (un valor por píxel).
            # Parámetros: 1024*784 + 784 = 803 600
            nn.Linear(1024, dim_imagen),
            # Tanh deja cada píxel en [-1, 1], igual que las imágenes reales
            # normalizadas en datos.py.
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [lote, 100]  ->  píxeles: [lote, 784]
        pixeles = self.red(z)
        # Se reacomoda el vector plano a imagen: [lote, 1, 28, 28]
        return pixeles.view(z.size(0), *self.forma)
