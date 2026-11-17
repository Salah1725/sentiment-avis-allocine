"""Télécharge le jeu Allociné depuis Hugging Face et l'enregistre dans data/.

Lancer depuis le dossier avis-allocine :
    .venv\\Scripts\\python telecharger_donnees.py
"""
import sys
from pathlib import Path

from datasets import load_dataset

dossier = Path(sys.argv[1] if len(sys.argv) > 1 else "data")
dossier.mkdir(exist_ok=True)

ds = load_dataset("tblard/allocine")
for split in ["train", "validation", "test"]:
    ds[split].to_pandas().to_parquet(dossier / f"{split}.parquet", index=False)
    print(f"{split} : {len(ds[split])} avis → {dossier / split}.parquet")
