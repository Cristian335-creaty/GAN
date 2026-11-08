"""
entrenamiento.py
================
Bucle de entrenamiento ADVERSARIAL.  En cada lote se hacen dos pasos:

  PASO 1 — Entrenar el DISCRIMINADOR (G queda fijo)
      x       = imágenes reales del lote
      G(z)    = imágenes falsas (se usa .detach() para no tocar G)
      L_D     = -log D(x) - log(1 - D(G(z)))
      optimizador_D.step()  -> actualiza SOLO los pesos de D

  PASO 2 — Entrenar el GENERADOR (D queda fijo)
      G(z)    = nuevas imágenes falsas
      L_G     = -log D(G(z))
      optimizador_G.step()  -> actualiza SOLO los pesos de G
"""

import csv

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import Config
from .datos import ruido_latente
from .perdidas import perdida_discriminador, perdida_generador
from .visualizacion import guardar_grilla


def entrenar(G: nn.Module, D: nn.Module, cargador: DataLoader, cfg: Config, dispositivo: torch.device) -> dict:
    # ---- OPTIMIZADORES (uno para cada red) --------------------------------
    # Cada optimizador recibe los parámetros de SU red: así, al dar un paso,
    # solo cambian los pesos de esa red.
    optim_G = torch.optim.Adam(G.parameters(), lr=cfg.lr_generador, betas=(cfg.beta1, cfg.beta2))
    optim_D = torch.optim.Adam(D.parameters(), lr=cfg.lr_discriminador, betas=(cfg.beta1, cfg.beta2))

    # Ruido FIJO: siempre las mismas z para ver cómo evoluciona G época a época.
    z_fijo = ruido_latente(cfg.num_muestras, cfg.dim_ruido, dispositivo)

    historial = {k: [] for k in ("perdida_D", "perdida_G", "perdida_D_real", "perdida_D_falsa", "D_x", "D_G_z")}
    carpeta = cfg.carpeta_resultados

    print("\n===== ENTRENAMIENTO =====")
    print(f"{'Época':>6} | {'L_D':>7} | {'L_G':>7} | {'D(x)':>6} | {'D(G(z))':>7}")
    print("-" * 46)

    for epoca in range(1, cfg.epocas + 1):
        acum = {k: 0.0 for k in historial}
        n_lotes = 0

        for (reales,) in cargador:
            reales = reales.to(dispositivo)
            b = reales.size(0)

            # ============ PASO 1: actualizar el DISCRIMINADOR ============
            z = ruido_latente(b, cfg.dim_ruido, dispositivo)
            falsas = G(z)

            d_real = D(reales)              # D(x)     -> debería tender a 1
            d_falso = D(falsas.detach())    # D(G(z))  -> debería tender a 0
            # .detach() corta el gradiente hacia G: en este paso solo aprende D.

            loss_D, loss_D_real, loss_D_falsa = perdida_discriminador(d_real, d_falso)

            optim_D.zero_grad()   # borra gradientes del paso anterior
            loss_D.backward()     # calcula dL_D / d(pesos de D)
            optim_D.step()        # actualiza los pesos de D

            # ============ PASO 2: actualizar el GENERADOR ============
            # Se vuelve a pasar G(z) por D (ya actualizado), esta vez SIN
            # detach, para que el gradiente fluya a través de D hasta G.
            d_falso_para_G = D(falsas)
            loss_G = perdida_generador(d_falso_para_G)

            optim_G.zero_grad()
            loss_G.backward()     # calcula dL_G / d(pesos de G)
            optim_G.step()        # actualiza SOLO los pesos de G

            # ---- acumular métricas del lote ----
            acum["perdida_D"] += loss_D.item()
            acum["perdida_G"] += loss_G.item()
            acum["perdida_D_real"] += loss_D_real.item()
            acum["perdida_D_falsa"] += loss_D_falsa.item()
            acum["D_x"] += d_real.mean().item()
            acum["D_G_z"] += d_falso.mean().item()
            n_lotes += 1

        # Promedio de la época
        for k in historial:
            historial[k].append(acum[k] / n_lotes)

        print(
            f"{epoca:>6} | {historial['perdida_D'][-1]:>7.4f} | {historial['perdida_G'][-1]:>7.4f} | "
            f"{historial['D_x'][-1]:>6.3f} | {historial['D_G_z'][-1]:>7.3f}"
        )

        # Guardar muestras periódicamente para ver la evolución
        if epoca % cfg.guardar_cada == 0 or epoca == 1 or epoca == cfg.epocas:
            G.eval()
            with torch.no_grad():
                muestras = G(z_fijo)
            G.train()
            guardar_grilla(
                muestras,
                carpeta / "evolucion" / f"muestras_epoca_{epoca:03d}.png",
                f"Época {epoca} — categoría: {cfg.nombre_categoria}",
            )

    # Guardar el historial completo en CSV (para abrirlo en Excel, etc.)
    with open(carpeta / "historial_entrenamiento.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoca", *historial.keys()])
        for i in range(cfg.epocas):
            w.writerow([i + 1, *(round(historial[k][i], 6) for k in historial)])

    return historial
