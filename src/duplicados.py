"""Auditoría de duplicados y agrupamiento por escena.

El Soil DataSet reúne fotografías de la web y cuadros de video: muchas imágenes
son copias exactas o casi exactas de otras, y varias de esas copias quedaron a
la vez en ``train`` y en ``test``. Evaluar así mide memoria, no reconocimiento.

Este módulo
1. detecta duplicados exactos por huella MD5 de los bytes del archivo, y
2. agrupa casi duplicados (misma escena) con una huella perceptual: la
   miniatura a color de 32×32 del recorte central, centrada y normalizada.
   Dos imágenes son «la misma escena» si la similitud coseno de sus huellas
   supera un umbral; los grupos son las componentes conexas de ese grafo
   (unión-búsqueda), de modo que la relación se vuelve transitiva.

Los grupos alimentan la validación cruzada estratificada por grupos, que nunca
deja una escena repartida entre entrenamiento y prueba.
"""
from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from .datos import leer, normalizar


def md5(ruta: str | Path) -> str:
    return hashlib.md5(Path(ruta).read_bytes()).hexdigest()


def duplicados_exactos(rutas: Sequence[Path]) -> list[list[int]]:
    """Índices agrupados por contenido binario idéntico (sólo grupos de ≥2)."""
    por_huella: dict[str, list[int]] = defaultdict(list)
    for i, ruta in enumerate(rutas):
        por_huella[md5(ruta)].append(i)
    return [g for g in por_huella.values() if len(g) > 1]


def huella_perceptual(img: np.ndarray, lado: int = 32) -> np.ndarray:
    """Miniatura a color centrada y de norma unitaria (vector de 3·lado² valores)."""
    t = normalizar(img, lado).astype(np.float32).ravel()
    t -= t.mean()
    return t / (np.linalg.norm(t) + 1e-12)


def matriz_similitud(rutas: Sequence[Path], lado: int = 32) -> np.ndarray:
    """Similitud coseno entre todas las huellas (diagonal = 1)."""
    H = np.stack([huella_perceptual(leer(r), lado) for r in rutas])
    return H @ H.T


def agrupar(similitud: np.ndarray, umbral: float) -> np.ndarray:
    """Componentes conexas del grafo «similitud > umbral» (unión-búsqueda).

    Devuelve un arreglo con el identificador de grupo de cada imagen,
    numerado de 0 en orden de primera aparición.
    """
    n = similitud.shape[0]
    padre = list(range(n))

    def raiz(a: int) -> int:
        while padre[a] != a:
            padre[a] = padre[padre[a]]  # compresión de caminos
            a = padre[a]
        return a

    filas, cols = np.where(np.triu(similitud > umbral, k=1))
    for i, j in zip(filas, cols):
        ri, rj = raiz(int(i)), raiz(int(j))
        if ri != rj:
            padre[ri] = rj

    renumerar: dict[int, int] = {}
    grupos = np.empty(n, dtype=int)
    for i in range(n):
        grupos[i] = renumerar.setdefault(raiz(i), len(renumerar))
    return grupos


def resumen_fuga(grupos: np.ndarray, particiones: Sequence[str]) -> dict:
    """Cuántas imágenes de prueba comparten escena con alguna de entrenamiento."""
    particiones = np.asarray(particiones)
    grupos_train = set(grupos[particiones == "train"])
    en_test = particiones == "test"
    contaminadas = np.array([g in grupos_train for g in grupos[en_test]])
    return {
        "imagenes_test": int(en_test.sum()),
        "test_con_gemela_en_train": int(contaminadas.sum()),
        "proporcion": float(contaminadas.mean()),
        "mascara_test_contaminada": contaminadas,
    }
