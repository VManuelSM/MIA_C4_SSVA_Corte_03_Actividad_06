"""Orquestación del experimento completo (usada por ``main.py`` y por el cuaderno).

Etapas
------
1. ``cargar``            — catálogo, lectura y normalización a 256×256.
2. ``auditar``           — duplicados exactos, casi duplicados y grupos de escena.
3. ``extraer``           — matrices de características de cada representación.
4. ``evaluar``           — protocolos A (partición original) y B (CV anidada por grupos).
5. ``ablaciones``        — sensibilidad a distancias, niveles de gris y resolución.
6. ``figuras``           — todas las figuras del reporte en ``salidas/``.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import visualizacion as vis
from .caracteristicas import ExtractorHaralick, ExtractorPixeles, matriz
from .datos import CLASES, NOMBRES, Muestra, catalogo, leer, normalizar
from .duplicados import agrupar, duplicados_exactos, matriz_similitud, resumen_fuga
from .evaluacion import linea_base, modelos, particion_original, validacion_anidada
from .haralick import ABREV_HARALICK, NOMBRES_HARALICK, ConfigHaralick

UMBRAL_ESCENA = 0.95
LADO = 256

#: Las cuatro representaciones del diseño factorial 2×2 (descriptor × color).
REPRESENTACIONES = {
    "Haralick · gris": ExtractorHaralick("gris", LADO, ConfigHaralick(32, (1, 2, 4, 8))),
    "Haralick · Lab": ExtractorHaralick("lab", LADO, ConfigHaralick(32, (1, 2, 4, 8))),
    "Píxeles · gris": ExtractorPixeles(32, color=False),
    "Píxeles · color": ExtractorPixeles(32, color=True),
}

#: Variantes para el análisis de sensibilidad (evaluadas con SVM-RBF).
ABLACIONES = {
    "Haralick · gris · d={1}": ExtractorHaralick("gris", LADO, ConfigHaralick(32, (1,))),
    "Haralick · gris · d={1,2,4,8} · 16 niveles": ExtractorHaralick("gris", LADO, ConfigHaralick(16, (1, 2, 4, 8))),
    "Haralick · gris · d={1,2,4,8} · 32 niveles": REPRESENTACIONES["Haralick · gris"],
    "Haralick · gris · d={1,2,4,8} · 64 niveles": ExtractorHaralick("gris", LADO, ConfigHaralick(64, (1, 2, 4, 8))),
    "Píxeles · color 8×8": ExtractorPixeles(8, color=True),
    "Píxeles · color 16×16": ExtractorPixeles(16, color=True),
    "Píxeles · color 32×32": REPRESENTACIONES["Píxeles · color"],
    "Píxeles · color 64×64": ExtractorPixeles(64, color=True),
}


@dataclass
class Experimento:
    raiz: Path
    salidas: Path
    repeticiones: int = 3
    muestras: list[Muestra] = field(default_factory=list)
    originales: list[np.ndarray] = field(default_factory=list)
    imagenes: list[np.ndarray] = field(default_factory=list)
    y: np.ndarray | None = None
    particiones: np.ndarray | None = None
    similitud: np.ndarray | None = None
    grupos: np.ndarray | None = None
    exactos: list[list[int]] = field(default_factory=list)
    fuga: dict = field(default_factory=dict)
    X: dict[str, np.ndarray] = field(default_factory=dict)
    cv: dict = field(default_factory=dict)
    original: dict = field(default_factory=dict)
    cv_sin_grupos: dict = field(default_factory=dict)
    ablacion: dict = field(default_factory=dict)
    tiempos: dict = field(default_factory=dict)

    # 1 ------------------------------------------------------------------
    def cargar(self) -> "Experimento":
        t = time.perf_counter()
        self.muestras = catalogo(self.raiz)
        self.originales = [leer(m.ruta) for m in self.muestras]
        self.imagenes = [normalizar(img, LADO) for img in self.originales]
        self.y = np.array([m.etiqueta for m in self.muestras])
        self.particiones = np.array([m.particion for m in self.muestras])
        self.tiempos["carga_s"] = time.perf_counter() - t
        return self

    def conteos(self) -> dict[str, dict[str, int]]:
        return {
            c: {p: int(((self.y == k) & (self.particiones == p)).sum()) for p in ("train", "test")}
            for k, c in enumerate(CLASES)
        }

    # 2 ------------------------------------------------------------------
    def auditar(self, umbral: float = UMBRAL_ESCENA) -> "Experimento":
        rutas = [m.ruta for m in self.muestras]
        self.exactos = duplicados_exactos(rutas)
        self.similitud = matriz_similitud(rutas)
        self.grupos = agrupar(self.similitud, umbral)
        self.fuga = resumen_fuga(self.grupos, self.particiones)
        return self

    def resumen_auditoria(self) -> dict:
        S = self.similitud.copy()
        np.fill_diagonal(S, -1)
        exactos_cruzados = [g for g in self.exactos if len(set(self.particiones[g])) > 1]
        por_clase = {}
        cont = self.fuga["mascara_test_contaminada"]
        te = np.where(self.particiones == "test")[0]
        for k, c in enumerate(CLASES):
            m = self.y[te] == k
            por_clase[c] = (int(cont[m].sum()), int(m.sum()))
        tam = np.bincount(self.grupos)
        return {
            "imagenes": len(self.muestras),
            "grupos_duplicado_exacto": len(self.exactos),
            "imagenes_en_duplicado_exacto": int(sum(len(g) for g in self.exactos)),
            "duplicados_exactos_entre_particiones": len(exactos_cruzados),
            "mediana_similitud_maxima": float(np.median(S.max(axis=1))),
            "umbral_escena": UMBRAL_ESCENA,
            "escenas": int(self.grupos.max() + 1),
            "escena_mas_grande": int(tam.max()),
            "escenas_por_clase": {
                c: int(len(set(self.grupos[self.y == k]))) for k, c in enumerate(CLASES)
            },
            "grupos_mixtos": int(sum(len(set(self.y[self.grupos == g])) > 1 for g in range(len(tam)))),
            "test_con_gemela_en_train": self.fuga["test_con_gemela_en_train"],
            "imagenes_test": self.fuga["imagenes_test"],
            "por_clase": por_clase,
        }

    # 3 ------------------------------------------------------------------
    def extraer(self, representaciones: dict | None = None) -> "Experimento":
        for nombre, ext in (representaciones or REPRESENTACIONES).items():
            if nombre in self.X:
                continue
            t = time.perf_counter()
            self.X[nombre] = matriz(ext, self.imagenes)
            self.tiempos[f"extraccion::{nombre}"] = time.perf_counter() - t
        return self

    # 4 ------------------------------------------------------------------
    def evaluar(self) -> "Experimento":
        cont = self.fuga["mascara_test_contaminada"]
        sin_grupos = np.arange(len(self.y))
        base = linea_base()
        rb = validacion_anidada(self.X["Haralick · gris"], self.y, self.grupos, base, "—", self.repeticiones)
        self.cv["Clase mayoritaria"] = rb
        for rep in REPRESENTACIONES:
            for mod in modelos():
                t = time.perf_counter()
                self.cv[(rep, mod.nombre)] = validacion_anidada(
                    self.X[rep], self.y, self.grupos, mod, rep, self.repeticiones)
                self.original[(rep, mod.nombre)] = particion_original(
                    self.X[rep], self.y, self.grupos, self.particiones, cont, mod)
                self.cv_sin_grupos[(rep, mod.nombre)] = validacion_anidada(
                    self.X[rep], self.y, sin_grupos, mod, rep, self.repeticiones)
                self.tiempos[f"evaluacion::{rep}::{mod.nombre}"] = time.perf_counter() - t
        return self

    def tabla_cv(self) -> list[dict]:
        filas = []
        for clave, r in self.cv.items():
            if clave == "Clase mayoritaria":
                continue
            rep, mod = clave
            o = self.original[clave]
            s = self.cv_sin_grupos[clave].resumen()
            filas.append({
                "representacion": rep, "modelo": mod, "rasgos": int(self.X[rep].shape[1]),
                **r.resumen(),
                "orig_exactitud": o["exactitud"], "orig_f1_macro": o["f1_macro"],
                "orig_exactitud_contaminadas": o["exactitud_contaminadas"],
                "orig_exactitud_limpias": o["exactitud_limpias"],
                "cv_sin_grupos_exactitud": s["exactitud"], "cv_sin_grupos_f1_macro": s["f1_macro"],
                "hiperparametros_frecuentes": _moda_hp(r.hiperparametros),
            })
        return filas

    def mejor(self, prefijo: str) -> tuple[str, str]:
        """(representación, modelo) con mayor F1 macro CV entre las que empiezan con ``prefijo``."""
        candidatos = [(k, v.resumen()["f1_macro"]) for k, v in self.cv.items()
                      if isinstance(k, tuple) and k[0].startswith(prefijo)]
        return max(candidatos, key=lambda kv: kv[1])[0]

    # 5 ------------------------------------------------------------------
    def ablaciones(self) -> "Experimento":
        self.extraer(ABLACIONES)
        svm = next(m for m in modelos() if m.nombre == "SVM-RBF")
        for nombre in ABLACIONES:
            r = validacion_anidada(self.X[nombre], self.y, self.grupos, svm, nombre, self.repeticiones)
            self.ablacion[nombre] = {"rasgos": int(self.X[nombre].shape[1]), **r.resumen()}
        return self

    # 6 ------------------------------------------------------------------
    def figuras(self) -> dict[str, Path]:
        return {**self.figuras_datos(), **self.figuras_resultados()}

    def figuras_datos(self) -> dict[str, Path]:
        """Figuras 1–9: datos, auditoría y representaciones (no requieren evaluar)."""
        s = self.salidas
        s.mkdir(parents=True, exist_ok=True)
        f: dict[str, Path] = {}
        f["fig01"] = vis.muestras_por_clase(self.imagenes, self.y, s / "fig01_muestras.png")

        # pares de fuga: test con su gemela más parecida en train
        te = np.where(self.particiones == "test")[0]
        tr = np.where(self.particiones == "train")[0]
        pares = []
        vistos = set()
        for clase_k in range(len(CLASES)):
            cand = [i for i in te if self.y[i] == clase_k]
            cand.sort(key=lambda i: -self.similitud[i, tr].max())
            for i in cand[:2]:
                j = tr[np.argmax(self.similitud[i, tr])]
                if j in vistos or self.similitud[i, j] <= UMBRAL_ESCENA:
                    continue
                vistos.add(j)
                pares.append((self.imagenes[i], self.imagenes[j], self.muestras[i].ident,
                              self.muestras[j].ident, float(self.similitud[i, j])))
        f["fig02"] = vis.pares_fuga(pares[:8], s / "fig02_pares_fuga.png")
        f["fig03"] = vis.fuga_por_clase(self.resumen_auditoria()["por_clase"], s / "fig03_fuga_por_clase.png")

        # Haralick: pipeline sobre una imagen rectangular de aluvial
        k_ej = next(i for i, m in enumerate(self.muestras)
                    if m.clase == "alluvial" and self.originales[i].shape[1] > 1.5 * self.originales[i].shape[0])
        f["fig04"] = vis.pipeline_haralick(self.originales[k_ej], self.imagenes[k_ej], 32,
                                           s / "fig04_pipeline_haralick.png")
        ejemplos = {}
        for k, c in enumerate(CLASES):
            idx = np.where(self.y == k)[0]
            # el más representativo: el más cercano a la media de su clase en Haralick·gris
            Xc = self.X["Haralick · gris"][idx]
            z = (Xc - Xc.mean(0)) / (Xc.std(0) + 1e-9)
            ejemplos[c] = self.imagenes[idx[np.argmin((z**2).sum(1))]]
        f["fig05"] = vis.glcm_por_clase(ejemplos, 32, s / "fig05_glcm_por_clase.png")
        sel = [1, 2, 8, 4, 5, 0]  # contraste, correlación, entropía, IDM, promedio de la suma, ASM (d=1, media)
        f["fig06"] = vis.cajas_haralick(self.X["Haralick · gris"], self.y, sel,
                                        [f"{NOMBRES_HARALICK[c]} · d=1" for c in sel],
                                        s / "fig06_cajas_haralick.png")
        # píxeles
        f["fig07"] = vis.pipeline_pixeles(self.imagenes[k_ej], 32, s / "fig07_pipeline_pixeles.png")
        f["fig08"] = vis.imagenes_medias(self.imagenes, self.y, 32, s / "fig08_imagenes_medias.png")
        # espacios de características
        f["fig09"] = vis.proyeccion_pca({
            "Haralick · Lab": self.X["Haralick · Lab"],
            "Píxeles · color": self.X["Píxeles · color"],
        }, self.y, s / "fig09_pca.png")
        return f

    def figuras_resultados(self) -> dict[str, Path]:
        """Figuras 10–15: requieren ``evaluar``."""
        s = self.salidas
        f: dict[str, Path] = {}
        tabla = self.tabla_cv()
        base = self.cv["Clase mayoritaria"].resumen()["f1_macro"]
        f["fig10"] = vis.barras_resultados(tabla, list(REPRESENTACIONES), [m.nombre for m in modelos()],
                                           base, s / "fig10_resultados_f1.png")
        filas_fuga = []
        for pref, eti in (("Haralick", "Con Haralick"), ("Píxeles", "Sin Haralick (píxeles)")):
            clave = self.mejor(pref)
            o, r = self.original[clave], self.cv[clave].resumen()
            filas_fuga.append({"etiqueta": f"{eti}\n{clave[0]} + {clave[1]}",
                               "orig_total": o["exactitud"], "orig_contaminadas": o["exactitud_contaminadas"],
                               "orig_limpias": o["exactitud_limpias"], "cv": r["exactitud"],
                               "cv_sin_grupos": self.cv_sin_grupos[clave].resumen()["exactitud"]})
        f["fig11"] = vis.efecto_fuga(filas_fuga, s / "fig11_efecto_fuga.png")
        casos = {}
        for pref, eti in (("Haralick", "Con Haralick"), ("Píxeles", "Sin Haralick")):
            clave = self.mejor(pref)
            casos[f"{eti}: {clave[0]} + {clave[1]}"] = (self.y, self.cv[clave].oof)
        f["fig12"] = vis.matrices_confusion(casos, s / "fig12_confusion.png")
        clave = self.mejor("Haralick")
        f["fig13"] = vis.errores(self.imagenes, self.y, self.cv[clave].oof, s / "fig13_errores_haralick.png")
        clave = self.mejor("Píxeles")
        f["fig14"] = vis.errores(self.imagenes, self.y, self.cv[clave].oof, s / "fig14_errores_pixeles.png")
        # importancias del bosque con Haralick·Lab
        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier(n_estimators=500, class_weight="balanced_subsample", random_state=0)
        rf.fit(self.X["Haralick · Lab"], self.y)
        f["fig15"] = vis.importancias(REPRESENTACIONES["Haralick · Lab"].nombres(), rf.feature_importances_,
                                      s / "fig15_importancias_haralick_lab.png")
        return f

    # ------------------------------------------------------------------
    def guardar_resultados(self) -> Path:
        out = {
            "conteos": self.conteos(),
            "auditoria": self.resumen_auditoria(),
            "linea_base": self.cv["Clase mayoritaria"].resumen(),
            "tabla": self.tabla_cv(),
            "ablaciones": self.ablacion,
            "mejor_haralick": list(self.mejor("Haralick")),
            "mejor_pixeles": list(self.mejor("Píxeles")),
            "tiempos": self.tiempos,
            "protocolo": {"repeticiones": self.repeticiones, "k_externo": 5, "k_interno": 3,
                          "umbral_escena": UMBRAL_ESCENA, "lado": LADO},
        }
        for pref in ("Haralick", "Píxeles"):
            clave = self.mejor(pref)
            from sklearn.metrics import classification_report
            out[f"reporte_{pref}"] = classification_report(
                self.y, self.cv[clave].oof, target_names=[NOMBRES[c] for c in CLASES],
                output_dict=True, zero_division=0)
        ruta = self.salidas / "resultados.json"
        ruta.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=_json))
        return ruta


def _moda_hp(hps: list[dict]) -> str:
    from collections import Counter
    if not hps or not hps[0]:
        return "—"
    c = Counter(json.dumps(h, sort_keys=True, default=str) for h in hps)
    hp, n = c.most_common(1)[0]
    d = json.loads(hp)
    txt = ", ".join(f"{k.replace('clf__', '')}={v}" for k, v in d.items())
    return f"{txt} ({n}/{len(hps)})"


def _json(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))
