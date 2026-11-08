# GAN explicada: Red Generadora vs. Red Discriminadora

Proyecto en Python (gestionado con **uv**) que entrena una GAN sencilla con
**PyTorch** sobre **una sola categoría** de MNIST (dígitos) o Fashion-MNIST
(ropa), y muestra paso a paso:

- los **parámetros** de la red Generadora y de la Discriminadora,
- las **funciones de pérdida** de cada una,
- los **valores** que producen G y D antes y después de entrenar,
- el **resultado de la GAN**, indicando la categoría usada.

Categoría por defecto: **MNIST, dígito 7**.

---

## 1. Instalación y ejecución (local con uv)

```powershell
cd gan-mnist
uv sync                      # crea .venv e instala torch, torchvision, matplotlib
uv run gan-mnist             # entrena con el dígito 7 durante 30 épocas
```

La primera vez descarga MNIST (~10 MB) en `./data`.

Otras opciones:

```powershell
uv run gan-mnist --categoria 3                    # dígito 3
uv run gan-mnist --dataset fashion --categoria 7  # Fashion-MNIST: Zapatilla
uv run gan-mnist --epocas 50 --lr-g 0.0002 --lr-d 0.0002
uv run gan-mnist --help
```

Tiempo aproximado en CPU: 5–15 s por época (unas 6 000 imágenes por categoría).

> **GPU (opcional):** el `torch` de PyPI en Windows es solo CPU. Si tienes
> NVIDIA y quieres usarla: `uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 --reinstall`.

---

## 2. ¿Dónde está cada cosa en el código?

| Archivo | Qué contiene |
|---|---|
| `src/gan_mnist/config.py` | Hiperparámetros y **categorías** disponibles |
| `src/gan_mnist/datos.py` | Descarga del dataset y **filtro por categoría** |
| `src/gan_mnist/generador.py` | **Se crea la Red Generadora** (clase `Generador`) |
| `src/gan_mnist/discriminador.py` | **Se crea la Red Discriminadora** (clase `Discriminador`) |
| `src/gan_mnist/perdidas.py` | **Funciones de pérdida** L_D y L_G |
| `src/gan_mnist/entrenamiento.py` | Bucle adversarial (paso de D, luego paso de G) |
| `src/gan_mnist/inspeccion.py` | Tabla de parámetros y valores de G y D |
| `src/gan_mnist/visualizacion.py` | Gráficas y grillas de imágenes |
| `src/gan_mnist/__init__.py` | `main()`: une todo. Aquí se **instancian** `G = Generador(...)` y `D = Discriminador(...)` |

---

## 3. Red Neuronal Generadora (G)

Recibe ruido aleatorio **z** (100 números ~ N(0,1)) y produce una imagen 28×28.

| Capa | Entrada → Salida | Activación | Parámetros |
|---|---|---|---|
| Linear 1 | 100 → 256 | LeakyReLU(0.2) | 25 856 |
| Linear 2 | 256 → 512 | BatchNorm + LeakyReLU | 131 584 + 1 024 |
| Linear 3 | 512 → 1024 | BatchNorm + LeakyReLU | 525 312 + 2 048 |
| Salida | 1024 → 784 | **Tanh** (píxeles en [-1, 1]) | 803 600 |
| **Total** | | | **1 489 424** |

## 4. Red Neuronal Discriminadora (D)

Recibe una imagen (real o falsa) y devuelve la probabilidad de que sea **real**.

| Capa | Entrada → Salida | Activación | Parámetros |
|---|---|---|---|
| Linear 1 | 784 → 512 | LeakyReLU(0.2) + Dropout(0.3) | 401 920 |
| Linear 2 | 512 → 256 | LeakyReLU(0.2) + Dropout(0.3) | 131 328 |
| Salida | 256 → 1 | **Sigmoid** (prob. en [0, 1]) | 257 |
| **Total** | | | **533 505** |

Parámetros de una capa Linear = entradas × salidas (pesos **W**) + salidas (sesgos **b**).

## 5. Funciones de pérdida

Ambas usan **Entropía Cruzada Binaria**: `BCE(p, y) = -[y·log p + (1-y)·log(1-p)]`

- **Discriminador:** `L_D = -log D(x) - log(1 - D(G(z)))`
  quiere D(x) → 1 (reales) y D(G(z)) → 0 (falsas).
- **Generador (no saturante):** `L_G = -log D(G(z))`
  quiere que D diga "real" (→ 1) a sus imágenes falsas.

Valores de referencia en el equilibrio teórico (D = 0.5 para todo):
`L_D = 2·ln2 ≈ 1.386`, `L_G = ln2 ≈ 0.693`.

Hiperparámetros: Adam, lr = 0.0002, β1 = 0.5, β2 = 0.999, lote = 64.

---

## 6. Qué se ve al ejecutar

En consola:
1. Tabla de parámetros de G y de D (forma, cantidad, media, desviación, mín., máx.).
2. Valores de muestra **antes** de entrenar: D(x), D(G(z)), rango de píxeles de G(z).
3. Por época: `L_D`, `L_G`, `D(x)`, `D(G(z))`.
4. Los mismos parámetros y valores **después** de entrenar.

En la carpeta `resultados/`:

| Archivo | Contenido |
|---|---|
| `resultado_final.png` | 64 imágenes generadas, con la **categoría usada** en el título |
| `reales_vs_generadas.png` | Reales de la categoría (arriba) vs. generadas (abajo) |
| `generador_sin_entrenar.png` | Lo que produce G antes de entrenar (ruido) |
| `evolucion/muestras_epoca_XXX.png` | Cómo mejora G con las épocas (mismo ruido fijo) |
| `perdidas.png` | Curvas L_D, L_G y las dos partes de L_D |
| `valores_D.png` | Evolución de D(x) y D(G(z)) |
| `histograma_pesos.png` | Distribución de los pesos de G y D |
| `parametros_*_inicial.txt` / `*_final.txt` | Tablas de parámetros |
| `historial_entrenamiento.csv` | Todas las métricas por época |
| `generador.pt`, `discriminador.pt` | Pesos entrenados |

### Cómo interpretar

- **D(x) alto y D(G(z)) bajo:** el Discriminador va ganando.
- **Ambos acercándose a 0.5:** G produce imágenes convincentes; D ya no distingue.
- **L_D → 0 y L_G muy alta:** D domina y G no aprende (bajar `--lr-d`).
- Una GAN no "converge" a una pérdida mínima como un clasificador: las dos
  pérdidas oscilan porque G y D compiten.
"# GAN" 
