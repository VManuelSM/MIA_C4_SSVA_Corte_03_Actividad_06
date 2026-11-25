"""Figuras del reporte. Cada función recibe datos ya calculados y guarda un PNG."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

from .datos import CLASES, COLORES, NOMBRES, recorte_cuadrado
from .haralick import DIRECCIONES, NOMBRES_HARALICK, cuantizar, glcm

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def _rgb(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def _guardar(fig, ruta: Path) -> Path:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta)
    plt.close(fig)
    return ruta


def _sin_ejes(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


# ---------------------------------------------------------------- datos

def muestras_por_clase(imgs: Sequence[np.ndarray], y: np.ndarray, ruta: Path,
                       por_clase: int = 6, semilla: int = 3) -> Path:
    rng = np.random.default_rng(semilla)
    fig, axes = plt.subplots(len(CLASES), por_clase, figsize=(1.6 * por_clase, 1.75 * len(CLASES)))
    for c, clase in enumerate(CLASES):
        idx = np.where(y == c)[0]
        elegidos = rng.choice(idx, por_clase, replace=False)
        for k, i in enumerate(elegidos):
            ax = axes[c, k]
            ax.imshow(_rgb(imgs[i]))
            _sin_ejes(ax)
        axes[c, 0].set_ylabel(f"{NOMBRES[clase]}\n(n = {len(idx)})", rotation=0,
                              ha="right", va="center", fontsize=10)
    fig.subplots_adjust(wspace=0.04, hspace=0.06)
    return _guardar(fig, ruta)


def pares_fuga(pares: Sequence[tuple[np.ndarray, np.ndarray, str, str, float]], ruta: Path,
               por_fila: int = 2) -> Path:
    """Cada par: (img_test, img_train, id_test, id_train, similitud)."""
    filas = int(np.ceil(len(pares) / por_fila))
    fig, axes = plt.subplots(filas, 2 * por_fila, figsize=(2.1 * 2 * por_fila, 2.35 * filas))
    axes = np.atleast_2d(axes)
    for k, (a, b, ia, ib, s) in enumerate(pares):
        f, c = divmod(k, por_fila)
        for j, (img, ident) in enumerate(((a, ia), (b, ib))):
            ax = axes[f, 2 * c + j]
            ax.imshow(_rgb(img)); _sin_ejes(ax)
            ax.set_title(ident, fontsize=7.5)
        etiqueta = "idéntica (MD5)" if s >= 0.99999 else f"similitud = {s:.3f}"
        axes[f, 2 * c + 1].set_xlabel(etiqueta, fontsize=8, color="#B5482F")
    fig.subplots_adjust(wspace=0.06, hspace=0.32)
    return _guardar(fig, ruta)


def fuga_por_clase(conteos: dict[str, tuple[int, int]], ruta: Path) -> Path:
    """conteos[clase] = (test con gemela en train, test total)."""
    fig, ax = plt.subplots(figsize=(6.2, 2.6))
    nombres = [NOMBRES[c] for c in CLASES]
    total = np.array([conteos[c][1] for c in CLASES])
    cont = np.array([conteos[c][0] for c in CLASES])
    ax.barh(nombres, total, color="#DDDDDD", label="Imágenes de prueba")
    ax.barh(nombres, cont, color=[COLORES[c] for c in CLASES],
            label="… con una gemela en entrenamiento")
    for i, (a, t) in enumerate(zip(cont, total)):
        ax.text(t + 0.6, i, f"{a}/{t}  ({100 * a / t:.0f} %)", va="center", fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Imágenes en «Soil Test»")
    ax.set_xlim(0, total.max() * 1.35)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    return _guardar(fig, ruta)


# ---------------------------------------------------------- Haralick

def pipeline_haralick(original: np.ndarray, normalizada: np.ndarray, niveles: int, ruta: Path) -> Path:
    gris = cv2.cvtColor(normalizada, cv2.COLOR_BGR2GRAY)
    q = cuantizar(gris, niveles)
    P = glcm(q, niveles, DIRECCIONES[0])
    alto, ancho = original.shape[:2]
    lado = min(alto, ancho)
    marcada = _rgb(original).copy()
    y0, x0 = (alto - lado) // 2, (ancho - lado) // 2
    cv2.rectangle(marcada, (x0, y0), (x0 + lado - 1, y0 + lado - 1), (255, 220, 0), max(2, lado // 90))
    # Dos renglones para que el texto sea legible a ancho de página:
    # (a) original a lo ancho + (b); abajo (c), (d) y (e).
    fig = plt.figure(figsize=(9.0, 6.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.32, wspace=0.28)
    paneles = [
        (fig.add_subplot(gs[0, :2]), marcada, f"(a) Original {ancho}×{alto} y recorte central", None),
        (fig.add_subplot(gs[0, 2]), _rgb(normalizada), "(b) Recorte 256×256 (INTER_AREA)", None),
        (fig.add_subplot(gs[1, 0]), gris, "(c) Luminancia Y (0–255)", "gray"),
        (fig.add_subplot(gs[1, 1]), q[96:160, 96:160], f"(d) Cuantizada a {niveles} niveles\n(acercamiento 64×64)", "gray"),
    ]
    for ax, im, tit, cmap in paneles:
        ax.imshow(im, cmap=cmap, interpolation="nearest"); _sin_ejes(ax); ax.set_title(tit, fontsize=10)
    ax_e = fig.add_subplot(gs[1, 2])
    im = ax_e.imshow(P + 1e-7, cmap="magma", norm=LogNorm(vmin=1e-5, vmax=P.max()))
    ax_e.set_title("(e) GLCM simétrica, d = 1, 0°\n(escala logarítmica)", fontsize=10)
    ax_e.set_xlabel("nivel j"); ax_e.set_ylabel("nivel i")
    fig.colorbar(im, ax=ax_e, fraction=0.046, pad=0.04)
    return _guardar(fig, ruta)


def glcm_por_clase(ejemplos: dict[str, np.ndarray], niveles: int, ruta: Path) -> Path:
    """Una imagen normalizada por clase: textura cuantizada y su GLCM media (d=1)."""
    fig, axes = plt.subplots(2, len(CLASES), figsize=(2.6 * len(CLASES), 5.4))
    for c, clase in enumerate(CLASES):
        gris = cv2.cvtColor(ejemplos[clase], cv2.COLOR_BGR2GRAY)
        q = cuantizar(gris, niveles)
        P = np.mean([glcm(q, niveles, d) for d in DIRECCIONES.values()], axis=0)
        axes[0, c].imshow(_rgb(ejemplos[clase])); _sin_ejes(axes[0, c])
        axes[0, c].set_title(NOMBRES[clase], color=COLORES[clase], fontweight="bold")
        axes[1, c].imshow(P + 1e-7, cmap="magma", norm=LogNorm(vmin=1e-5, vmax=0.05))
        axes[1, c].set_xticks([0, niveles - 1]); axes[1, c].set_yticks([0, niveles - 1])
        dispersion = (P * (np.subtract.outer(np.arange(niveles), np.arange(niveles)) ** 2)).sum()
        axes[1, c].set_xlabel(f"contraste = {dispersion:.2f}", fontsize=9)
    axes[1, 0].set_ylabel("GLCM media, d = 1\n(escala log)")
    fig.subplots_adjust(wspace=0.15, hspace=0.12)
    return _guardar(fig, ruta)


def cajas_haralick(X: np.ndarray, y: np.ndarray, columnas: Sequence[int], titulos: Sequence[str],
                   ruta: Path, ncols: int = 3) -> Path:
    filas = int(np.ceil(len(columnas) / ncols))
    fig, axes = plt.subplots(filas, ncols, figsize=(3.3 * ncols, 2.7 * filas))
    axes = np.atleast_1d(axes).ravel()
    for ax, col, tit in zip(axes, columnas, titulos):
        datos = [X[y == c, col] for c in range(len(CLASES))]
        bp = ax.boxplot(datos, patch_artist=True, widths=0.6, showfliers=False,
                        medianprops={"color": "white", "linewidth": 1.5})
        for caja, clase in zip(bp["boxes"], CLASES):
            caja.set_facecolor(COLORES[clase]); caja.set_edgecolor(COLORES[clase])
        for k, d in enumerate(datos):
            jit = np.random.default_rng(k).uniform(-0.18, 0.18, len(d))
            ax.scatter(np.full(len(d), k + 1) + jit, d, s=3, color="#555555", alpha=0.35, zorder=3)
        ax.set_xticks(range(1, len(CLASES) + 1), [NOMBRES[c].split(" ")[0] for c in CLASES], fontsize=8.5)
        ax.set_title(tit)
    for ax in axes[len(columnas):]:
        ax.set_visible(False)
    fig.tight_layout()
    return _guardar(fig, ruta)


# ----------------------------------------------------------- píxeles

def pipeline_pixeles(normalizada: np.ndarray, lado: int, ruta: Path) -> Path:
    peque = cv2.resize(normalizada, (lado, lado), interpolation=cv2.INTER_AREA)
    vector = peque.reshape(-1, 3)  # BGR por píxel, orden de renglón
    # Dos renglones: arriba las tres imágenes; abajo el vector a todo lo ancho.
    fig = plt.figure(figsize=(9.0, 4.6))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 0.13, 0.55], width_ratios=[1, 1, 2.2],
                          hspace=0.42, wspace=0.12)
    ax0 = fig.add_subplot(gs[0, 0]); ax0.imshow(_rgb(normalizada)); _sin_ejes(ax0)
    ax0.set_title("(a) Recorte 256×256", fontsize=10)
    ax1 = fig.add_subplot(gs[0, 1]); ax1.imshow(_rgb(peque), interpolation="nearest"); _sin_ejes(ax1)
    ax1.set_title(f"(b) Reducida a {lado}×{lado}\n(INTER_AREA)", fontsize=10)
    ax2 = fig.add_subplot(gs[0, 2])
    canales = np.hstack([peque[..., 2], peque[..., 1], peque[..., 0]])
    ax2.imshow(canales, cmap="gray", interpolation="nearest"); _sin_ejes(ax2)
    ax2.set_title("(c) Canales R | G | B", fontsize=10)
    n = 96
    ax3 = fig.add_subplot(gs[1, :])
    tira = _rgb(peque).reshape(1, -1, 3)[:, :n]
    ax3.imshow(tira, aspect="auto", interpolation="nearest"); _sin_ejes(ax3)
    ax3.set_title(rf"(d) Vector $\mathbf{{x}}\in[0,1]^{{{3 * lado * lado}}}$: primeros {n} de {lado * lado} píxeles "
                  "como tira de color (arriba) y como valores (abajo)", fontsize=10)
    ax4 = fig.add_subplot(gs[2, :])
    vals = vector[:n, ::-1].ravel() / 255.0
    ax4.plot(vals, lw=0.8, color="#333333")
    ax4.set_xlim(0, len(vals) - 1); ax4.set_ylim(0, 1)
    ax4.set_xlabel("componente del vector (R, G, B intercalados)", fontsize=9.5)
    ax4.set_ylabel("valor", fontsize=9.5)
    return _guardar(fig, ruta)


def imagenes_medias(imgs: Sequence[np.ndarray], y: np.ndarray, lado: int, ruta: Path) -> Path:
    fig, axes = plt.subplots(1, len(CLASES), figsize=(2.4 * len(CLASES), 2.7))
    for c, clase in enumerate(CLASES):
        pila = np.stack([cv2.resize(imgs[i], (lado, lado), interpolation=cv2.INTER_AREA)
                         for i in np.where(y == c)[0]]).astype(np.float64)
        media = pila.mean(axis=0).astype(np.uint8)
        axes[c].imshow(_rgb(media), interpolation="nearest"); _sin_ejes(axes[c])
        axes[c].set_title(f"{NOMBRES[clase]}\n(media de {len(pila)})", color=COLORES[clase], fontweight="bold")
    fig.subplots_adjust(wspace=0.08)
    return _guardar(fig, ruta)


# --------------------------------------------------------- resultados

def proyeccion_pca(espacios: dict[str, np.ndarray], y: np.ndarray, ruta: Path) -> Path:
    fig, axes = plt.subplots(1, len(espacios), figsize=(4.6 * len(espacios), 4.0))
    for ax, (titulo, X) in zip(np.atleast_1d(axes), espacios.items()):
        Z = StandardScaler().fit_transform(X)
        pca = PCA(n_components=2, random_state=0).fit(Z)
        P = pca.transform(Z)
        for c, clase in enumerate(CLASES):
            m = y == c
            ax.scatter(P[m, 0], P[m, 1], s=12, alpha=0.75, color=COLORES[clase],
                       label=NOMBRES[clase], edgecolors="white", linewidths=0.3)
        v = pca.explained_variance_ratio_ * 100
        ax.set_xlabel(f"CP1 ({v[0]:.0f} % var.)"); ax.set_ylabel(f"CP2 ({v[1]:.0f} % var.)")
        ax.set_title(f"{titulo}  ·  {X.shape[1]} rasgos")
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=8, markerscale=1.5)
    fig.tight_layout()
    return _guardar(fig, ruta)


def barras_resultados(tabla: list[dict], representaciones: Sequence[str], modelos: Sequence[str],
                      base: float, ruta: Path, metrica: str = "f1_macro",
                      etiqueta: str = "F1 macro (CV anidada por grupos)") -> Path:
    fig, ax = plt.subplots(figsize=(8.6, 3.6))
    ancho = 0.8 / len(modelos)
    paleta = ["#4C72B0", "#DD8452", "#55A868"]
    x = np.arange(len(representaciones))
    for k, mod in enumerate(modelos):
        med = [next(t[metrica] for t in tabla if t["representacion"] == r and t["modelo"] == mod)
               for r in representaciones]
        de = [next(t[metrica + "_de"] for t in tabla if t["representacion"] == r and t["modelo"] == mod)
              for r in representaciones]
        barras = ax.bar(x + (k - (len(modelos) - 1) / 2) * ancho, med, ancho * 0.92, yerr=de,
                        capsize=2.5, color=paleta[k], label=mod, error_kw={"lw": 0.8})
        for b, v, e in zip(barras, med, de):
            ax.text(b.get_x() + b.get_width() / 2, v + e + 0.015, f"{v:.2f}", ha="center", fontsize=7)
    ax.axhline(base, ls="--", lw=1, color="#999999")
    ax.text(len(representaciones) - 0.5, base + 0.015, "clase mayoritaria", ha="right",
            fontsize=8, color="#777777")
    ax.set_xticks(x, representaciones, fontsize=9)
    ax.set_ylim(0, 1.1); ax.set_ylabel(etiqueta)
    ax.legend(frameon=False, ncol=len(modelos), fontsize=8.5, loc="upper center",
              bbox_to_anchor=(0.5, 1.13))
    return _guardar(fig, ruta)


def efecto_fuga(filas: list[dict], ruta: Path) -> Path:
    """filas: {etiqueta, orig_total, orig_contaminadas, orig_limpias, cv} (exactitud)."""
    fig, ax = plt.subplots(figsize=(8.6, 3.4))
    claves = [("orig_contaminadas", "Partición original: prueba con gemela en entrenamiento", "#B5482F"),
              ("orig_total", "Partición original: toda la prueba", "#E3A587"),
              ("orig_limpias", "Partición original: prueba sin gemela", "#9DB4CF"),
              ("cv_sin_grupos", "CV anidada sin grupos (imágenes al azar)", "#7F7F7F"),
              ("cv", "CV anidada por grupos (honesta)", "#2F5D8A")]
    x = np.arange(len(filas)); ancho = 0.16
    for k, (clave, eti, col) in enumerate(claves):
        vals = [f[clave] for f in filas]
        b = ax.bar(x + (k - 2) * ancho, vals, ancho * 0.92, color=col, label=eti)
        for bb, v in zip(b, vals):
            ax.text(bb.get_x() + bb.get_width() / 2, v + 0.015, f"{v:.2f}", ha="center", fontsize=7)
    ax.set_xticks(x, [f["etiqueta"] for f in filas], fontsize=9)
    ax.set_ylim(0, 1.12); ax.set_ylabel("Exactitud")
    ax.legend(frameon=False, fontsize=7.5, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.27))
    return _guardar(fig, ruta)


def matrices_confusion(casos: dict[str, tuple[np.ndarray, np.ndarray]], ruta: Path) -> Path:
    fig, axes = plt.subplots(1, len(casos), figsize=(4.3 * len(casos), 3.9))
    etiquetas = [NOMBRES[c].split(" ")[0] for c in CLASES]
    for ax, (titulo, (y, pred)) in zip(np.atleast_1d(axes), casos.items()):
        cm = confusion_matrix(y, pred, labels=range(len(CLASES)))
        norm = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(len(CLASES)):
            for j in range(len(CLASES)):
                ax.text(j, i, f"{cm[i, j]}\n{100 * norm[i, j]:.0f} %", ha="center", va="center",
                        fontsize=8, color="white" if norm[i, j] > 0.55 else "#222222")
        ax.set_xticks(range(len(CLASES)), etiquetas, fontsize=8.5)
        ax.set_yticks(range(len(CLASES)), etiquetas, fontsize=8.5)
        ax.set_xlabel("Predicha"); ax.set_ylabel("Real")
        ax.set_title(titulo, fontsize=9.5)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.tight_layout()
    return _guardar(fig, ruta)


def errores(imgs: Sequence[np.ndarray], y: np.ndarray, pred: np.ndarray, ruta: Path,
            maximo: int = 10, titulo_extra: Sequence[str] | None = None) -> Path:
    idx = np.where(pred != y)[0][:maximo]
    cols = 5
    filas = int(np.ceil(len(idx) / cols))
    fig, axes = plt.subplots(filas, cols, figsize=(2.2 * cols, 2.55 * filas))
    axes = np.atleast_1d(axes).ravel()
    for ax, i in zip(axes, idx):
        ax.imshow(_rgb(imgs[i])); _sin_ejes(ax)
        ax.set_title(f"real: {NOMBRES[CLASES[y[i]]].split(' ')[0]}\npred.: {NOMBRES[CLASES[pred[i]]].split(' ')[0]}",
                     fontsize=8.5)
    for ax in axes[len(idx):]:
        ax.set_visible(False)
    fig.subplots_adjust(wspace=0.06, hspace=0.38)
    return _guardar(fig, ruta)


def importancias(nombres: Sequence[str], valores: np.ndarray, ruta: Path, top: int = 15) -> Path:
    orden = np.argsort(valores)[::-1][:top]
    fig, ax = plt.subplots(figsize=(6.4, 0.26 * top + 0.8))
    ax.barh([nombres[i] for i in orden][::-1], valores[orden][::-1], color="#2F5D8A")
    ax.set_xlabel("Importancia (disminución media de impureza de Gini)")
    ax.tick_params(axis="y", labelsize=8)
    return _guardar(fig, ruta)
