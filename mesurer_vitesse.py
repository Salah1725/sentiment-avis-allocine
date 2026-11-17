"""Mesure la vitesse et l'accuracy des modèles sur les mêmes 500 avis que le notebook 4.

Deux vitesses différentes :
- latence : un avis à la fois, comme dans l'API quand un utilisateur envoie un avis ;
- débit   : beaucoup d'avis d'un coup (traitement par paquets), comme pour classer
            une grosse base d'avis.

Lancer depuis le dossier avis-allocine :
    .venv\\Scripts\\python mesurer_vitesse.py
Pour Qwen, on relit data/qwen_500.csv (notebook 4) au lieu de tout relancer.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd

import modeles

N_LATENCE = 100      # avis classés un par un pour mesurer la latence
TAILLE_PAQUET = 32   # taille des paquets pour CamemBERT


def mesurer(nom, fonction_proba, textes, labels):
    # Échauffement : le premier appel charge le modèle, on ne le chronomètre pas.
    fonction_proba(textes[:1])

    latences = []
    for texte in textes[:N_LATENCE]:
        debut = time.perf_counter()
        fonction_proba([texte])
        latences.append(time.perf_counter() - debut)

    debut = time.perf_counter()
    probas = np.concatenate([fonction_proba(textes[i:i + TAILLE_PAQUET])
                             for i in range(0, len(textes), TAILLE_PAQUET)])
    duree = time.perf_counter() - debut

    return {
        "modele": nom,
        "accuracy_500": ((probas >= 0.5).astype(int) == labels).mean(),
        "latence_ms": 1000 * np.median(latences),
        "avis_par_seconde": len(textes) / duree,
    }


def main():
    val = pd.read_parquet("data/validation.parquet")
    echantillon = val.sample(500, random_state=42).reset_index(drop=True)
    textes, labels = echantillon["review"].tolist(), echantillon["label"].to_numpy()

    lignes = []
    for nom, fonction in [("tfidf", modeles.proba_tfidf), ("camembert", modeles.proba_camembert)]:
        try:
            print(f"Mesure de {nom}...")
            lignes.append(mesurer(nom, fonction, textes, labels))
        except modeles.ModeleIndisponible as e:
            print(f"  ignoré : {e}")

    fichier_qwen = Path("data/qwen_500.csv")
    if fichier_qwen.exists():
        qwen = pd.read_csv(fichier_qwen)
        lignes.append({
            "modele": "qwen",
            "accuracy_500": (qwen["prediction"].to_numpy() == labels).mean(),
            "latence_ms": 1000 * qwen["secondes"].median(),
            "avis_par_seconde": 1 / qwen["secondes"].mean(),  # un avis à la fois
        })

    resultats = pd.DataFrame(lignes).round({"accuracy_500": 3, "latence_ms": 1, "avis_par_seconde": 1})
    Path("resultats").mkdir(exist_ok=True)
    resultats.to_csv("resultats/vitesse.csv", index=False)
    print()
    print(resultats.to_string(index=False))
    print("\nRésultats enregistrés dans resultats/vitesse.csv")


if __name__ == "__main__":
    main()
