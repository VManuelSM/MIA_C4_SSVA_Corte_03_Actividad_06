#!/usr/bin/env bash
# Descarga el Soil DataSet (Kaggle: naivedatamodel/soil-dataset, v2, GPL-3) y lo
# reorganiza en datos/{train,test}/{alluvial,black,cinder,red}/*.jpg
set -euo pipefail
cd "$(dirname "$0")"
tmp=$(mktemp -d)
curl -sSL -o "$tmp/soil.zip" "https://www.kaggle.com/api/v1/datasets/download/naivedatamodel/soil-dataset"
python - "$tmp/soil.zip" datos <<'PY'
import os, sys, zipfile
z = zipfile.ZipFile(sys.argv[1]); out = sys.argv[2]
mapa = {"Soil Train/Soil Train": "train", "Soil Test/Soil Test": "test"}
for info in z.infolist():
    if info.is_dir():
        continue
    carpeta, nombre = os.path.split(info.filename)
    particion, clase = os.path.split(carpeta)
    assert particion in mapa and nombre.lower().endswith(".jpg")
    destino = os.path.join(out, mapa[particion], clase.replace(" Soil", "").lower())
    os.makedirs(destino, exist_ok=True)
    with open(os.path.join(destino, nombre), "wb") as fh:
        fh.write(z.read(info))
print("Imágenes:", sum(len(f) for _, _, f in os.walk(out)))
PY
rm -rf "$tmp"
