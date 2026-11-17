"""Chargement des trois modèles et prédiction sur un avis.

Utilisé par l'API (api.py), la démo (app.py) et la mesure de vitesse
(mesurer_vitesse.py), pour que les trois utilisent exactement le même code.

- tfidf     : TF-IDF + bigrammes + régression logistique (models/tfidf.joblib)
- camembert : CamemBERT fine-tuné (dossier local ou identifiant Hugging Face,
              variable d'environnement CAMEMBERT_PATH)
- qwen      : Qwen 2.5 3B en zero-shot via Ollama (Ollama doit être lancé)
"""
import json
import os
import time
from functools import lru_cache
from pathlib import Path

TFIDF_PATH = Path("models/tfidf.joblib")
CAMEMBERT_PATH = os.getenv("CAMEMBERT_PATH", "models/camembert-allocine")
QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen2.5:3b")
MAX_TOKENS_CAMEMBERT = 256

CONSIGNE = (
    "Tu classes des critiques de films en français, publiées sur Allociné. "
    "Indique si l'avis global de l'auteur sur le film est positif ou négatif. "
    "Juge le verdict de l'auteur dans son ensemble, pas des phrases isolées."
)
SCHEMA = {
    "type": "object",
    "properties": {"sentiment": {"type": "string", "enum": ["positif", "negatif"]}},
    "required": ["sentiment"],
    "additionalProperties": False,
}

NOMS = {
    "tfidf": "TF-IDF + bigrammes",
    "camembert": "CamemBERT fine-tuné",
    "qwen": "Qwen 2.5 3B (zero-shot)",
}


class ModeleIndisponible(RuntimeError):
    """Le modèle demandé n'est pas installé ou pas lancé."""


# ---------- Chargement (une seule fois, grâce au cache) ----------

@lru_cache(maxsize=1)
def charger_tfidf():
    import joblib

    if not TFIDF_PATH.exists():
        raise ModeleIndisponible("models/tfidf.joblib absent : lance entrainer_tfidf.py")
    return joblib.load(TFIDF_PATH)


@lru_cache(maxsize=1)
def charger_camembert():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    # Soit un dossier local, soit un identifiant Hugging Face du type "compte/nom".
    est_local = CAMEMBERT_PATH.startswith(("models", ".", "/")) or ":" in CAMEMBERT_PATH
    if est_local and not Path(CAMEMBERT_PATH).exists():
        raise ModeleIndisponible(f"CamemBERT introuvable dans {CAMEMBERT_PATH}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(CAMEMBERT_PATH)
        modele = AutoModelForSequenceClassification.from_pretrained(CAMEMBERT_PATH)
    except OSError as e:
        raise ModeleIndisponible(f"CamemBERT introuvable : {e}") from e
    modele.eval()
    return tokenizer, modele


def qwen_disponible():
    import ollama

    try:
        return any(m.model == QWEN_MODEL for m in ollama.list().models)
    except Exception:
        return False


def disponibles():
    """Liste des modèles utilisables maintenant."""
    resultat = []
    for nom, test in [("tfidf", charger_tfidf), ("camembert", charger_camembert)]:
        try:
            test()
            resultat.append(nom)
        except ModeleIndisponible:
            pass
    if qwen_disponible():
        resultat.append("qwen")
    return resultat


# ---------- Prédiction ----------

def proba_tfidf(textes):
    """Probabilité « positif » pour une liste d'avis."""
    return charger_tfidf().predict_proba(list(textes))[:, 1]


def proba_camembert(textes):
    """Probabilité « positif » pour une liste d'avis (traités en un seul paquet)."""
    import torch

    tokenizer, modele = charger_camembert()
    entrees = tokenizer(list(textes), return_tensors="pt", truncation=True,
                        max_length=MAX_TOKENS_CAMEMBERT, padding=True)
    with torch.no_grad():
        logits = modele(**entrees).logits
    return torch.softmax(logits, dim=-1)[:, 1].numpy()


def sentiment_qwen(texte):
    """1 si Qwen juge l'avis positif, 0 sinon."""
    import ollama

    try:
        reponse = ollama.chat(
            model=QWEN_MODEL,
            messages=[{"role": "system", "content": CONSIGNE},
                      {"role": "user", "content": texte}],
            format=SCHEMA,
            options={"temperature": 0},
        )
    except Exception as e:
        raise ModeleIndisponible(f"Ollama indisponible (est-il lancé ?) : {e}") from e
    return int(json.loads(reponse.message.content)["sentiment"] == "positif")


def predire(texte, modele):
    """Classe un avis avec le modèle choisi et mesure le temps de réponse."""
    debut = time.perf_counter()
    if modele == "tfidf":
        proba = float(proba_tfidf([texte])[0])
    elif modele == "camembert":
        proba = float(proba_camembert([texte])[0])
    elif modele == "qwen":
        proba = None  # un LLM donne une réponse, pas une probabilité
        etiquette = sentiment_qwen(texte)
    else:
        raise ValueError(f"Modèle inconnu : {modele}")
    secondes = time.perf_counter() - debut

    if proba is not None:
        etiquette = int(proba >= 0.5)
    return {
        "modele": modele,
        "sentiment": "positif" if etiquette == 1 else "negatif",
        "proba_positif": proba,
        "secondes": round(secondes, 4),
    }
