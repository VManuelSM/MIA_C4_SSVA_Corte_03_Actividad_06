"""Reproduce el experimento completo: auditoría, características, evaluación y figuras.

Uso::

    python -m src.main            # 3 repeticiones de la CV externa (≈ 3–5 min)
    python -m src.main --rapido   # 1 repetición, para probar
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

from .experimento import Experimento

RAIZ = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datos", type=Path, default=RAIZ / "datos")
    p.add_argument("--salidas", type=Path, default=RAIZ / "salidas")
    p.add_argument("--rapido", action="store_true")
    a = p.parse_args()
    warnings.filterwarnings("ignore", category=UserWarning)

    exp = Experimento(a.datos, a.salidas, repeticiones=1 if a.rapido else 3)
    print("1/6 Carga y normalización…", flush=True)
    exp.cargar()
    print("2/6 Auditoría de duplicados…", flush=True)
    exp.auditar()
    au = exp.resumen_auditoria()
    print(f"    {au['escenas']} escenas; {au['test_con_gemela_en_train']}/{au['imagenes_test']} "
          "imágenes de prueba con gemela en entrenamiento", flush=True)
    print("3/6 Extracción de características…", flush=True)
    exp.extraer()
    print("4/6 Evaluación (protocolos A y B)…", flush=True)
    exp.evaluar()
    for fila in exp.tabla_cv():
        print(f"    {fila['representacion']:18s} {fila['modelo']:17s} "
              f"F1={fila['f1_macro']:.3f}±{fila['f1_macro_de']:.3f}  "
              f"exactitud={fila['exactitud']:.3f}  (partición original: {fila['orig_exactitud']:.3f})",
              flush=True)
    print("5/6 Ablaciones…", flush=True)
    exp.ablaciones()
    print("6/6 Figuras y resultados…", flush=True)
    exp.salidas.mkdir(parents=True, exist_ok=True)
    exp.figuras()
    print("    →", exp.guardar_resultados())


if __name__ == "__main__":
    main()
