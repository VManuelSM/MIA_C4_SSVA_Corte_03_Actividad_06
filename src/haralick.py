r"""Matriz de co-ocurrencia de niveles de gris (GLCM) y descriptores de Haralick.

Implementación propia en NumPy, validada contra ``mahotas`` (ver
``tests/test_haralick.py``).

Definiciones (Haralick, Shanmugam y Dinstein, 1973). Sea :math:`P(i,j)` la GLCM
simétrica y normalizada de :math:`L` niveles para un desplazamiento
:math:`(\Delta y, \Delta x)`: la probabilidad de que un píxel de nivel
:math:`i` tenga, a ese desplazamiento, un vecino de nivel :math:`j`. Con las
marginales :math:`p_x(i)=\sum_j P(i,j)`, :math:`p_y(j)=\sum_i P(i,j)` y las
distribuciones de suma y diferencia

.. math::
    p_{x+y}(k) = \sum_{i+j=k} P(i,j), \qquad
    p_{x-y}(k) = \sum_{|i-j|=k} P(i,j),

las 13 características son

==  ==================================  =========================================
f1  Segundo momento angular (energía)    :math:`\sum P^2`
f2  Contraste                            :math:`\sum_k k^2\,p_{x-y}(k)`
f3  Correlación                          :math:`(\sum ij P - \mu_x\mu_y)/(\sigma_x\sigma_y)`
f4  Suma de cuadrados: varianza          :math:`\sigma_x^2`
f5  Momento de diferencia inversa        :math:`\sum P/(1+(i-j)^2)`
f6  Promedio de la suma                  :math:`\sum_k k\,p_{x+y}(k)`
f7  Varianza de la suma                  :math:`\sum_k (k-f_6)^2 p_{x+y}(k)`
f8  Entropía de la suma                  :math:`-\sum p_{x+y}\log_2 p_{x+y}`
f9  Entropía                             :math:`-\sum P\log_2 P`
f10 Varianza de la diferencia            :math:`\mathrm{Var}[p_{x-y}]`
f11 Entropía de la diferencia            :math:`-\sum p_{x-y}\log_2 p_{x-y}`
f12 Medida de información de corr. 1     :math:`(f_9-H_{XY1})/\max(H_X,H_Y)`
f13 Medida de información de corr. 2     :math:`\sqrt{1-e^{-2(H_{XY2}-f_9)}}`
==  ==================================  =========================================

con :math:`H_{XY1}=-\sum P\log_2(p_x p_y)` y
:math:`H_{XY2}=-\sum p_x p_y\log_2(p_x p_y)`. La f7 usa :math:`f_6` (no
:math:`f_8`, errata del artículo original) y la f10 es la varianza de la
distribución :math:`p_{x-y}`, como en el artículo.

Para volver los descriptores casi invariantes a la rotación, se calculan en las
cuatro direcciones (0°, 45°, 90° y 135°) y se resumen, como propone Haralick,
con la **media** y el **rango** (máximo − mínimo) entre direcciones.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

#: Desplazamientos unitarios (Δy, Δx) por ángulo, en coordenadas de imagen.
DIRECCIONES: dict[int, tuple[int, int]] = {
    0: (0, 1),
    45: (-1, 1),
    90: (-1, 0),
    135: (-1, -1),
}

NOMBRES_HARALICK: tuple[str, ...] = (
    "Energía (ASM)",
    "Contraste",
    "Correlación",
    "Varianza",
    "Homogeneidad (IDM)",
    "Promedio de la suma",
    "Varianza de la suma",
    "Entropía de la suma",
    "Entropía",
    "Varianza de la diferencia",
    "Entropía de la diferencia",
    "Info. de correlación 1",
    "Info. de correlación 2",
)

ABREV_HARALICK: tuple[str, ...] = (
    "ASM", "CON", "COR", "VAR", "IDM", "SAVG", "SVAR",
    "SENT", "ENT", "DVAR", "DENT", "IMC1", "IMC2",
)


def cuantizar(gris: np.ndarray, niveles: int) -> np.ndarray:
    """Reduce una imagen de 8 bits a ``niveles`` niveles uniformes (0…niveles−1).

    Con 256 niveles la GLCM de una imagen de 256×256 tiene 65 536 celdas para
    ~65 000 pares de píxeles: casi toda vacía y muy ruidosa. Cuantizar a 32
    niveles la vuelve densa y estable sin perder la estructura de la textura.
    """
    if gris.dtype != np.uint8:
        raise TypeError("Se espera una imagen uint8")
    return ((gris.astype(np.int32) * niveles) >> 8).astype(np.int32)


def glcm(
    img: np.ndarray,
    niveles: int,
    desplazamiento: tuple[int, int],
    simetrica: bool = True,
    normalizada: bool = True,
) -> np.ndarray:
    """GLCM de ``img`` (enteros 0…niveles−1) para un desplazamiento (Δy, Δx).

    Cuenta los pares (img[y, x], img[y+Δy, x+Δx]) dentro de la imagen con
    ``np.bincount`` sobre el índice lineal ``a·L + b``; la versión simétrica
    suma la transpuesta (cuenta también el par en sentido inverso).
    """
    dy, dx = desplazamiento
    alto, ancho = img.shape
    y0, y1 = max(0, -dy), alto - max(0, dy)
    x0, x1 = max(0, -dx), ancho - max(0, dx)
    a = img[y0:y1, x0:x1]
    b = img[y0 + dy : y1 + dy, x0 + dx : x1 + dx]
    conteo = np.bincount(
        (a * niveles + b).ravel(), minlength=niveles * niveles
    ).reshape(niveles, niveles).astype(np.float64)
    if simetrica:
        conteo = conteo + conteo.T
    if normalizada:
        conteo /= conteo.sum()
    return conteo


def _entropia(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def caracteristicas_haralick(P: np.ndarray) -> np.ndarray:
    """Las 13 características de Haralick de una GLCM normalizada ``P``."""
    L = P.shape[0]
    k = np.arange(L, dtype=np.float64)
    i, j = np.indices((L, L))

    px = P.sum(axis=1)
    py = P.sum(axis=0)
    mux, muy = k @ px, k @ py
    varx = (k**2) @ px - mux**2
    vary = (k**2) @ py - muy**2

    p_suma = np.bincount((i + j).ravel(), weights=P.ravel(), minlength=2 * L - 1)
    p_dif = np.bincount(np.abs(i - j).ravel(), weights=P.ravel(), minlength=L)
    ks = np.arange(2 * L - 1, dtype=np.float64)

    f = np.empty(13)
    f[0] = (P**2).sum()
    f[1] = (k**2) @ p_dif
    sxy = np.sqrt(varx * vary)
    f[2] = ((i * j * P).sum() - mux * muy) / sxy if sxy > 0 else 1.0
    f[3] = varx
    f[4] = (P / (1.0 + (i - j) ** 2)).sum()
    f[5] = ks @ p_suma
    f[6] = ((ks - f[5]) ** 2) @ p_suma
    f[7] = _entropia(p_suma)
    f[8] = _entropia(P)
    f[9] = (k**2) @ p_dif - (k @ p_dif) ** 2
    f[10] = _entropia(p_dif)

    hx, hy = _entropia(px), _entropia(py)
    pxpy = np.outer(px, py)
    nz = pxpy > 0
    hxy1 = float(-(P[nz] * np.log2(pxpy[nz])).sum())
    hxy2 = _entropia(pxpy)
    hmax = max(hx, hy)
    f[11] = (f[8] - hxy1) / hmax if hmax > 0 else f[8] - hxy1
    f[12] = np.sqrt(max(0.0, 1.0 - np.exp(-2.0 * (hxy2 - f[8]))))
    return f


def haralick_direcciones(img_q: np.ndarray, niveles: int, distancia: int = 1) -> np.ndarray:
    """Matriz 4×13: las 13 características en cada una de las 4 direcciones."""
    return np.stack(
        [
            caracteristicas_haralick(
                glcm(img_q, niveles, (dy * distancia, dx * distancia))
            )
            for dy, dx in DIRECCIONES.values()
        ]
    )


@dataclass(frozen=True)
class ConfigHaralick:
    """Parámetros del descriptor de Haralick de una imagen (o de un canal)."""

    niveles: int = 32
    distancias: Sequence[int] = (1,)

    @property
    def dimension(self) -> int:
        return 26 * len(self.distancias)

    def nombres(self, prefijo: str = "") -> list[str]:
        nombres = []
        for d in self.distancias:
            for estad in ("media", "rango"):
                nombres += [f"{prefijo}{a}_d{d}_{estad}" for a in ABREV_HARALICK]
        return nombres


def descriptor_haralick(canal_u8: np.ndarray, cfg: ConfigHaralick) -> np.ndarray:
    """Vector [media(13) | rango(13)] por distancia, concatenado."""
    q = cuantizar(canal_u8, cfg.niveles)
    partes: list[np.ndarray] = []
    for d in cfg.distancias:
        M = haralick_direcciones(q, cfg.niveles, d)
        partes += [M.mean(axis=0), M.max(axis=0) - M.min(axis=0)]
    return np.concatenate(partes)


def lote(canales: Iterable[np.ndarray], cfg: ConfigHaralick) -> np.ndarray:
    return np.stack([descriptor_haralick(c, cfg) for c in canales])
