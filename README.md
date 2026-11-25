# MIA_C4_SSVA_Corte_03_Actividad_06

**Reconocimiento de suelos con y sin descriptores de textura de Haralick**

| | |
|---|---|
| **Alumno** | Víctor Manuel Santos Martínez |
| **Matrícula** | 253220020 |
| **Materia** | Sistemas de Visión Artificial |
| **Profesor** | Dr. Ernesto García Amaro |
| **Actividad** | Actividad 6 — Reconocimiento de suelos, Corte 3 |
| **Universidad** | Universidad Politécnica Metropolitana de Hidalgo |
| **Fecha de entrega** | 25 de noviembre de 2026 |

---

## Descripción de la actividad

Se reconoce el tipo de suelo en fotografías del
[Soil DataSet](https://www.kaggle.com/datasets/naivedatamodel/soil-dataset)
(368 imágenes de cuatro clases: aluvial, negro, escoria volcánica —*cinder*— y
rojo) con dos experimentos que sólo difieren en la representación de la
imagen:

1. **Con Haralick.** Se extraen descriptores de textura de la matriz de
   co-ocurrencia de niveles de gris (GLCM): las 13 características de Haralick
   en cuatro direcciones, resumidas con media y rango, a las distancias
   1, 2, 4 y 8 px, en escala de grises y en cada canal de CIE-Lab.
2. **Sin Haralick.** Se usan como características los propios píxeles de la
   imagen reducida a 32×32, en gris y a color.

Ambas representaciones pasan por los mismos tres clasificadores (k-NN,
SVM con kernel RBF y bosque aleatorio) y el mismo protocolo de evaluación.

### Hallazgo previo: el conjunto filtra información entre entrenamiento y prueba

Antes de clasificar se auditó el conjunto. **50 de las 73 imágenes de
`Soil Test` tienen una gemela, exacta o casi exacta, en `Soil Train`**: hay
copias byte a byte y cuadros consecutivos de un mismo video. Con esa
partición, cualquier clasificador acierta el 100 % de las imágenes
«gemelas», y la exactitud reportada mide en parte memoria. Por eso el
resultado principal se obtiene con **validación cruzada anidada estratificada
por grupos de escena** (5 pliegues externos × 3 repeticiones, 3 internos para
los hiperparámetros), donde ninguna escena cae a la vez en entrenamiento y
prueba. La partición original se reporta como referencia, desglosada.

## Estructura del repositorio

```
├── reconocimiento_suelos.ipynb  # Cuaderno ejecutado: teoría, experimento y resultados
├── src/
│   ├── datos.py                 # Catálogo, lectura y normalización 256×256
│   ├── duplicados.py            # MD5, huella perceptual, grupos de escena, fuga
│   ├── haralick.py              # GLCM y 13 descriptores de Haralick (NumPy)
│   ├── caracteristicas.py       # Extractores (patrón Estrategia): Haralick y píxeles
│   ├── evaluacion.py            # Clasificadores y protocolos A/B
│   ├── experimento.py           # Orquestación de etapas
│   ├── visualizacion.py         # Figuras del reporte
│   └── main.py                  # Reproduce todo
├── tests/test_haralick.py       # Valida la implementación contra mahotas
├── salidas/                     # Figuras (figNN_*.png) y resultados.json
├── descargar_datos.sh           # Descarga y reorganiza el dataset en datos/
└── requirements.txt
```

El dataset (licencia GPL-3) no se versiona: `descargar_datos.sh` lo baja de
Kaggle y lo deja en `datos/{train,test}/{alluvial,black,cinder,red}/`.

## Reproducir

```bash
pip install -r requirements.txt
./descargar_datos.sh
python -m pytest -q tests        # GLCM/Haralick propia == mahotas
python -m src.main               # ≈ 5 min; figuras y resultados en salidas/
# o el cuaderno completo:
jupyter nbconvert --to notebook --execute --inplace reconocimiento_suelos.ipynb
```
