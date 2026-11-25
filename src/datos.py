"""Carga del Soil DataSet y normalización geométrica común a ambos experimentos.

El conjunto de Kaggle (naivedatamodel/soil-dataset, v2) trae 368 fotografías
JPEG de cuatro tipos de suelo repartidas en dos carpetas, ``Soil Train`` y
``Soil Test``. Al descomprimirlo se reorganiza como::

    datos/
      train/{alluvial,black,cinder,red}/*.jpg
      test/{alluvial,black,cinder,red}/*.jpg

Las imágenes vienen de la web: tamaños de 100×100 a 2560×1920 px, relaciones
de aspecto distintas y, en algunas, marcas de agua, manos o cielo. Por eso todo
experimento parte de la misma normalización: recorte cuadrado central y
redimensión por área (``cv2.INTER_AREA``), que promedia en lugar de muestrear y
evita el aliasing al reducir.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

#: Orden canónico de las clases (índice = etiqueta numérica).
CLASES: tuple[str, ...] = ("alluvial", "black", "cinder", "red")

#: Nombres en español para tablas y figuras.
NOMBRES: dict[str, str] = {
    "alluvial": "Aluvial",
    "black": "Negro",
    "cinder": "Escoria (cinder)",
    "red": "Rojo",
}

#: Color de cada clase en las figuras (coherente con el suelo que representa).
COLORES: dict[str, str] = {
    "alluvial": "#C9A86A",
    "black": "#3A3A3A",
    "cinder": "#7A6F9B",
    "red": "#B5482F",
}


@dataclass(frozen=True)
class Muestra:
    """Una imagen del conjunto con su partición original y su clase."""

    ruta: Path
    particion: str  # "train" | "test" (partición que trae el dataset)
    clase: str

    @property
    def etiqueta(self) -> int:
        return CLASES.index(self.clase)

    @property
    def ident(self) -> str:
        return f"{self.particion}/{self.clase}/{self.ruta.name}"


def catalogo(raiz: str | Path) -> list[Muestra]:
    """Enumera las imágenes en un orden determinista (partición, clase, número)."""
    raiz = Path(raiz)
    muestras: list[Muestra] = []
    for particion in ("train", "test"):
        for clase in CLASES:
            carpeta = raiz / particion / clase
            rutas = sorted(carpeta.glob("*.jpg"), key=lambda p: int(p.stem))
            muestras.extend(Muestra(r, particion, clase) for r in rutas)
    if not muestras:
        raise FileNotFoundError(
            f"No hay imágenes en {raiz}. Descargue el dataset (ver README)."
        )
    return muestras


def leer(ruta: str | Path) -> np.ndarray:
    """Lee una imagen BGR de 8 bits (admite rutas con acentos o espacios)."""
    datos = np.fromfile(str(ruta), dtype=np.uint8)
    img = cv2.imdecode(datos, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"No se pudo decodificar {ruta}")
    return img


def recorte_cuadrado(img: np.ndarray, fraccion: float = 1.0) -> np.ndarray:
    """Recorta el cuadrado central de lado ``fraccion · min(alto, ancho)``.

    Descarta los márgenes donde suelen caer cielo, bordes blancos y marcas de
    agua, y fija la relación de aspecto 1:1 sin deformar la textura.
    """
    alto, ancho = img.shape[:2]
    lado = int(round(min(alto, ancho) * fraccion))
    y0 = (alto - lado) // 2
    x0 = (ancho - lado) // 2
    return img[y0 : y0 + lado, x0 : x0 + lado]


def normalizar(img: np.ndarray, lado: int, fraccion: float = 1.0) -> np.ndarray:
    """Recorte cuadrado central + redimensión por área a ``lado × lado``."""
    cuadrado = recorte_cuadrado(img, fraccion)
    interp = cv2.INTER_AREA if cuadrado.shape[0] >= lado else cv2.INTER_CUBIC
    return cv2.resize(cuadrado, (lado, lado), interpolation=interp)
