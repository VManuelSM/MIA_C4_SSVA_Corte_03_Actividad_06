"""Clasificadores y protocolos de evaluación.

Se comparan tres clasificadores clásicos con supuestos distintos sobre el
espacio de características:

* **k-NN** — no paramétrico, decide por cercanía euclidiana; sensible a la
  escala y a la dimensión.
* **SVM con kernel RBF** — margen máximo en un espacio de rasgos implícito;
  robusta en alta dimensión.
* **Bosque aleatorio** — ensamble de árboles; invariante a la escala de cada
  rasgo y con importancia de variables.

Todos van dentro de un ``Pipeline`` con ``StandardScaler`` para que la
estandarización se ajuste sólo con los datos de entrenamiento de cada pliegue
(si se ajustara con todo el conjunto, la prueba «contaminaría» la media y la
desviación).

Protocolos
----------
A. **Partición original** del dataset (``Soil Train`` → ``Soil Test``). Es la
   que usaría cualquiera que descargue el conjunto, pero comparte escenas
   entre entrenamiento y prueba: mide, en parte, memoria.
B. **Validación cruzada anidada estratificada por grupos** sobre las 368
   imágenes. El pliegue externo (5 pliegues, 3 repeticiones) estima el
   desempeño; el interno (3 pliegues) elige los hiperparámetros. Ninguna
   escena (grupo de casi duplicados) cae a la vez en entrenamiento y prueba.
   Es la estimación honesta y la que se reporta como resultado principal.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


@dataclass
class Modelo:
    nombre: str
    estimador: Pipeline
    rejilla: dict


def modelos(semilla: int = 0) -> list[Modelo]:
    return [
        Modelo(
            "k-NN",
            Pipeline([("esc", StandardScaler()), ("clf", KNeighborsClassifier())]),
            {"clf__n_neighbors": [1, 3, 5, 7, 9], "clf__weights": ["uniform", "distance"]},
        ),
        Modelo(
            "SVM-RBF",
            Pipeline([
                ("esc", StandardScaler()),
                ("clf", SVC(kernel="rbf", class_weight="balanced")),
            ]),
            {"clf__C": [0.1, 1, 10, 100], "clf__gamma": ["scale", 1e-3, 1e-2, 1e-1]},
        ),
        Modelo(
            "Bosque aleatorio",
            Pipeline([
                ("esc", StandardScaler()),
                ("clf", RandomForestClassifier(
                    n_estimators=200, class_weight="balanced_subsample",
                    random_state=semilla, n_jobs=1,
                )),
            ]),
            {"clf__max_features": ["sqrt", 0.2], "clf__min_samples_leaf": [1, 3]},
        ),
    ]


def linea_base() -> Modelo:
    """Clasificador trivial: siempre predice la clase mayoritaria (aluvial)."""
    return Modelo("Clase mayoritaria", Pipeline([("clf", DummyClassifier(strategy="most_frequent"))]), {})


def metricas(y_true, y_pred) -> dict[str, float]:
    return {
        "exactitud": accuracy_score(y_true, y_pred),
        "exactitud_balanceada": balanced_accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def _ajustar(modelo: Modelo, X, y, grupos, semilla: int, k_int: int, n_jobs: int):
    if not modelo.rejilla:
        return clone(modelo.estimador).fit(X, y), {}
    busq = GridSearchCV(
        clone(modelo.estimador),
        modelo.rejilla,
        cv=StratifiedGroupKFold(n_splits=k_int, shuffle=True, random_state=semilla),
        scoring="f1_macro",
        n_jobs=n_jobs,
        refit=True,
    )
    busq.fit(X, y, groups=grupos)
    return busq.best_estimator_, busq.best_params_


@dataclass
class ResultadoCV:
    modelo: str
    representacion: str
    pliegues: list[dict] = field(default_factory=list)
    oof: np.ndarray | None = None  # predicción fuera de pliegue (1.ª repetición)
    hiperparametros: list[dict] = field(default_factory=list)

    def resumen(self) -> dict[str, float]:
        out = {}
        for m in ("exactitud", "exactitud_balanceada", "f1_macro"):
            v = np.array([p[m] for p in self.pliegues])
            out[m] = float(v.mean())
            out[m + "_de"] = float(v.std(ddof=1))
        return out


def validacion_anidada(
    X: np.ndarray,
    y: np.ndarray,
    grupos: np.ndarray,
    modelo: Modelo,
    representacion: str,
    repeticiones: int = 3,
    k_ext: int = 5,
    k_int: int = 3,
    n_jobs: int = -1,
) -> ResultadoCV:
    """Protocolo B: CV anidada estratificada por grupos, repetida."""
    res = ResultadoCV(modelo.nombre, representacion)
    res.oof = np.full(len(y), -1)
    for r in range(repeticiones):
        externo = StratifiedGroupKFold(n_splits=k_ext, shuffle=True, random_state=r)
        for pl, (tr, te) in enumerate(externo.split(X, y, grupos)):
            assert not set(grupos[tr]) & set(grupos[te]), "fuga de grupo"
            est, hp = _ajustar(modelo, X[tr], y[tr], grupos[tr], r, k_int, n_jobs)
            pred = est.predict(X[te])
            res.pliegues.append({"repeticion": r, "pliegue": pl, **metricas(y[te], pred)})
            res.hiperparametros.append(hp)
            if r == 0:
                res.oof[te] = pred
    return res


def particion_original(
    X: np.ndarray,
    y: np.ndarray,
    grupos: np.ndarray,
    particiones: Sequence[str],
    contaminadas_test: np.ndarray,
    modelo: Modelo,
    k_int: int = 3,
    n_jobs: int = -1,
) -> dict:
    """Protocolo A: entrena en ``train`` y prueba en ``test`` tal como vienen.

    Además del total, desglosa el desempeño en las imágenes de prueba que
    tienen una gemela en entrenamiento y en las que no.
    """
    particiones = np.asarray(particiones)
    tr = np.where(particiones == "train")[0]
    te = np.where(particiones == "test")[0]
    est, hp = _ajustar(modelo, X[tr], y[tr], grupos[tr], 0, k_int, n_jobs)
    pred = est.predict(X[te])
    acierto = pred == y[te]
    return {
        "modelo": modelo.nombre,
        "hiperparametros": hp,
        **metricas(y[te], pred),
        "exactitud_contaminadas": float(acierto[contaminadas_test].mean()),
        "exactitud_limpias": float(acierto[~contaminadas_test].mean()),
        "n_contaminadas": int(contaminadas_test.sum()),
        "n_limpias": int((~contaminadas_test).sum()),
        "prediccion": pred,
    }
