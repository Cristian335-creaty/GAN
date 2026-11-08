"""
GAN (Generative Adversarial Network) explicada paso a paso
==========================================================

Punto de entrada:  uv run gan-mnist   (opciones: uv run gan-mnist --help)

Flujo del programa:
  1. Leer opciones (categoría, épocas, ...)          -> config.py
  2. Cargar SOLO las imágenes de la categoría elegida -> datos.py
  3. CREAR la red Generadora                          -> generador.py
  4. CREAR la red Discriminadora                      -> discriminador.py
  5. Mostrar parámetros y valores ANTES de entrenar   -> inspeccion.py
  6. Entrenar con las funciones de pérdida            -> entrenamiento.py / perdidas.py
  7. Mostrar parámetros y valores DESPUÉS de entrenar -> inspeccion.py
  8. Guardar gráficas y el resultado final de la GAN  -> visualizacion.py
"""

import argparse
import random
from pathlib import Path

import torch

from .config import CATEGORIAS, Config
from .datos import cargar_datos, ruido_latente
from .discriminador import Discriminador
from .entrenamiento import entrenar
from .generador import Generador
from .inspeccion import resumen_parametros, valores_muestra
from .visualizacion import (
    comparar_reales_falsas,
    graficar_perdidas,
    graficar_valores_D,
    guardar_grilla,
    histograma_pesos,
)


def leer_argumentos() -> Config:
    cfg = Config()
    p = argparse.ArgumentParser(description="Entrena una GAN sobre UNA categoría de MNIST / Fashion-MNIST.")
    p.add_argument("--dataset", choices=["mnist", "fashion"], default=cfg.dataset)
    p.add_argument("--categoria", type=int, choices=range(10), default=cfg.categoria,
                   help="0..9. MNIST: el dígito. Fashion: 0=Camiseta 1=Pantalón 2=Suéter 3=Vestido "
                        "4=Abrigo 5=Sandalia 6=Camisa 7=Zapatilla 8=Bolso 9=Botín")
    p.add_argument("--epocas", type=int, default=cfg.epocas)
    p.add_argument("--tam-lote", type=int, default=cfg.tam_lote)
    p.add_argument("--lr-g", type=float, default=cfg.lr_generador, help="tasa de aprendizaje del Generador")
    p.add_argument("--lr-d", type=float, default=cfg.lr_discriminador, help="tasa de aprendizaje del Discriminador")
    p.add_argument("--dim-ruido", type=int, default=cfg.dim_ruido)
    p.add_argument("--semilla", type=int, default=cfg.semilla)
    p.add_argument("--resultados", type=Path, default=cfg.carpeta_resultados)
    a = p.parse_args()

    cfg.dataset, cfg.categoria, cfg.epocas = a.dataset, a.categoria, a.epocas
    cfg.tam_lote, cfg.lr_generador, cfg.lr_discriminador = a.tam_lote, a.lr_g, a.lr_d
    cfg.dim_ruido, cfg.semilla, cfg.carpeta_resultados = a.dim_ruido, a.semilla, a.resultados
    return cfg


def main() -> None:
    cfg = leer_argumentos()

    # Semillas -> resultados reproducibles
    random.seed(cfg.semilla)
    torch.manual_seed(cfg.semilla)

    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    carpeta = cfg.carpeta_resultados
    carpeta.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(" GAN — Red Generadora vs. Red Discriminadora")
    print("=" * 60)
    print(f" Dispositivo      : {dispositivo}")
    print(f" Dataset          : {cfg.dataset.upper()}")
    print(f" CATEGORÍA USADA  : {cfg.categoria} = '{cfg.nombre_categoria}'")
    print(f" Épocas / lote    : {cfg.epocas} / {cfg.tam_lote}")
    print(f" lr G / lr D      : {cfg.lr_generador} / {cfg.lr_discriminador}")
    print("=" * 60)

    # ---------------- 2. DATOS ----------------
    cargador = cargar_datos(cfg)

    # ==================================================================
    # 3. CREACIÓN DE LA RED NEURONAL GENERADORA  (ver generador.py)
    #    Entrada: ruido z (100)  ->  Salida: imagen 1x28x28
    # ==================================================================
    G = Generador(dim_ruido=cfg.dim_ruido, dim_imagen=cfg.dim_imagen,
                  forma=(1, cfg.alto, cfg.ancho)).to(dispositivo)

    # ==================================================================
    # 4. CREACIÓN DE LA RED NEURONAL DISCRIMINADORA  (ver discriminador.py)
    #    Entrada: imagen 1x28x28  ->  Salida: probabilidad de ser real
    # ==================================================================
    D = Discriminador(dim_imagen=cfg.dim_imagen).to(dispositivo)

    # ---------------- 5. PARÁMETROS Y VALORES ANTES ----------------
    resumen_parametros(G, "GENERADOR (antes de entrenar)", carpeta / "parametros_generador_inicial.txt")
    resumen_parametros(D, "DISCRIMINADOR (antes de entrenar)", carpeta / "parametros_discriminador_inicial.txt")

    reales_muestra = next(iter(cargador))[0][:16].to(dispositivo)
    z_muestra = ruido_latente(16, cfg.dim_ruido, dispositivo)
    valores_muestra(G, D, reales_muestra, z_muestra, "ANTES de entrenar")
    with torch.no_grad():
        G.eval()
        guardar_grilla(G(ruido_latente(cfg.num_muestras, cfg.dim_ruido, dispositivo)),
                       carpeta / "generador_sin_entrenar.png",
                       "Generador SIN entrenar (solo ruido)")
        G.train()

    # ---------------- 6. ENTRENAMIENTO ----------------
    historial = entrenar(G, D, cargador, cfg, dispositivo)

    # ---------------- 7. PARÁMETROS Y VALORES DESPUÉS ----------------
    resumen_parametros(G, "GENERADOR (entrenado)", carpeta / "parametros_generador_final.txt")
    resumen_parametros(D, "DISCRIMINADOR (entrenado)", carpeta / "parametros_discriminador_final.txt")
    valores_muestra(G, D, reales_muestra, z_muestra, "DESPUÉS de entrenar")

    # ---------------- 8. RESULTADO DE LA GAN ----------------
    G.eval()
    with torch.no_grad():
        finales = G(ruido_latente(cfg.num_muestras, cfg.dim_ruido, dispositivo))
        comparacion = G(z_muestra)
    titulo = f"Resultado de la GAN — {cfg.dataset.upper()}, categoría usada: {cfg.categoria} ({cfg.nombre_categoria})"
    guardar_grilla(finales, carpeta / "resultado_final.png", titulo)
    comparar_reales_falsas(reales_muestra, comparacion, carpeta / "reales_vs_generadas.png", titulo)
    graficar_perdidas(historial, carpeta / "perdidas.png", cfg.nombre_categoria)
    graficar_valores_D(historial, carpeta / "valores_D.png", cfg.nombre_categoria)
    histograma_pesos(G, D, carpeta / "histograma_pesos.png")

    # Guardar los pesos entrenados (se pueden recargar con torch.load)
    torch.save(G.state_dict(), carpeta / "generador.pt")
    torch.save(D.state_dict(), carpeta / "discriminador.pt")

    print("=" * 60)
    print(f" Listo. Categoría usada: {cfg.categoria} = '{cfg.nombre_categoria}'")
    print(f" Resultados guardados en: {carpeta.resolve()}")
    print("   resultado_final.png, reales_vs_generadas.png, perdidas.png,")
    print("   valores_D.png, histograma_pesos.png, evolucion/, parametros_*.txt,")
    print("   historial_entrenamiento.csv, generador.pt, discriminador.pt")
    print("=" * 60)


__all__ = ["main", "CATEGORIAS"]
