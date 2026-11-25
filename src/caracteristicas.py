"""Extractores de características (patrón Estrategia).

Los dos experimentos comparten todo el pipeline —normalización geométrica,
clasificadores, protocolo de evaluación— y difieren sólo en *cómo* se convierte
una imagen en un vector. Cada extractor encapsula esa decisión detrás de la
misma interfaz (``extraer`` / ``nombres``), de modo que el resto del código no
sabe ni le importa qué representación recibe.

* :class:`ExtractorHaralick` — **con Haralick**: 13 descriptores de textura de
  la GLCM por dirección, resumidos con media y rango, a varias distancias, en
  escala de grises o en cada canal de CIE-Lab.
* :class:`ExtractorPixeles` — **sin Haralick**: los propios píxeles de la
  imagen reducida, aplanados en un vector (en gris o a color).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence

import cv2
import numpy as np

from .datos import normalizar
from .haralick import ConfigHaralick, descriptor_haralick


class Extractor(Protocol):
    nombre: str

    def extraer(self, img_bgr: np.ndarray) -> np.ndarray: ...

    def nombres(self) -> list[str]: ...


def canales(img_bgr: np.ndarray, espacio: str) -> list[tuple[str, np.ndarray]]:
    """Canales de 8 bits sobre los que se mide la textura."""
    if espacio == "gris":
        return [("Y", cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY))]
    if espacio == "lab":
        L, a, b = cv2.split(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB))
        return [("L", L), ("a", a), ("b", b)]
    raise ValueError(f"Espacio no soportado: {espacio}")


@dataclass
class ExtractorHaralick:
    """Descriptor de textura de Haralick sobre la imagen normalizada."""

    espacio: str = "gris"
    lado: int = 256
    config: ConfigHaralick = field(default_factory=lambda: ConfigHaralick(32, (1, 2, 4, 8)))

    @property
    def nombre(self) -> str:
        d = ",".join(map(str, self.config.distancias))
        return f"Haralick {self.espacio} (L={self.config.niveles}, d={{{d}}})"

    def extraer(self, img_bgr: np.ndarray) -> np.ndarray:
        img = normalizar(img_bgr, self.lado)
        return np.concatenate(
            [descriptor_haralick(c, self.config) for _, c in canales(img, self.espacio)]
        )

    def nombres(self) -> list[str]:
        ids = ["Y"] if self.espacio == "gris" else ["L", "a", "b"]
        return [n for c in ids for n in self.config.nombres(prefijo=f"{c}:")]


@dataclass
class ExtractorPixeles:
    """Los píxeles de la imagen reducida a ``lado × lado`` como vector."""

    lado: int = 32
    color: bool = True

    @property
    def nombre(self) -> str:
        return f"Píxeles {'color' if self.color else 'gris'} {self.lado}×{self.lado}"

    def extraer(self, img_bgr: np.ndarray) -> np.ndarray:
        img = normalizar(img_bgr, self.lado)
        if not self.color:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img.astype(np.float32).ravel() / 255.0

    def nombres(self) -> list[str]:
        if self.color:
            return [f"{c}[{y},{x}]" for y in range(self.lado) for x in range(self.lado) for c in "BGR"]
        return [f"Y[{y},{x}]" for y in range(self.lado) for x in range(self.lado)]


def matriz(extractor: Extractor, imagenes: Sequence[np.ndarray]) -> np.ndarray:
    """Aplica el extractor a todas las imágenes → matriz (n_imágenes × n_rasgos)."""
    return np.stack([extractor.extraer(img) for img in imagenes]).astype(np.float64)
