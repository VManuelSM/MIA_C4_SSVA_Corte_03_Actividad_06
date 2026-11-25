"""Valida la implementación propia de GLCM/Haralick contra mahotas.

mahotas recorre las direcciones (0,1), (1,1), (1,0), (1,-1); con GLCM simétrica
equivalen a nuestras 0°, 135°, 90° y 45°. Se usa ``use_x_minus_y_variance=True``
para que la f10 siga la definición del artículo original.
"""
import numpy as np
import pytest

mahotas = pytest.importorskip("mahotas")

from src.haralick import cuantizar, haralick_direcciones

ORDEN_MAHOTAS = [0, 3, 2, 1]  # nuestras [0°, 45°, 90°, 135°] en índices de mahotas


@pytest.mark.parametrize("semilla", [0, 1, 2])
@pytest.mark.parametrize("distancia", [1, 2, 4])
def test_coincide_con_mahotas(semilla, distancia):
    rng = np.random.default_rng(semilla)
    # textura sintética: ruido suavizado + gradiente, cuantizada a 32 niveles
    base = rng.normal(size=(96, 96)).cumsum(axis=1)
    base = (255 * (base - base.min()) / np.ptp(base)).astype(np.uint8)
    q = cuantizar(base, 32)
    niveles = int(q.max()) + 1
    propio = haralick_direcciones(q, niveles, distancia)
    ref = mahotas.features.haralick(
        q.astype(np.uint8), distance=distancia, use_x_minus_y_variance=True
    )[ORDEN_MAHOTAS]
    np.testing.assert_allclose(propio, ref, rtol=1e-9, atol=1e-12)
