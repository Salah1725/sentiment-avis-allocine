"""Entraîne le modèle TF-IDF + bigrammes et le sauvegarde dans models/tfidf.joblib.

Lancer depuis le dossier avis-allocine :
    .venv\\Scripts\\python entrainer_tfidf.py
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

train = pd.read_parquet("data/train.parquet")
val = pd.read_parquet("data/validation.parquet")

# Le pipeline enchaîne le vectoriseur et le modèle : fit/predict s'appliquent
# directement au texte brut, et un seul fichier contient les deux.
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=200_000)),
    ("modele", LogisticRegression(max_iter=1000)),
])

pipeline.fit(train["review"], train["label"])

accuracy = accuracy_score(val["label"], pipeline.predict(val["review"]))
print(f"Accuracy validation : {accuracy:.4f}")

Path("models").mkdir(exist_ok=True)
joblib.dump(pipeline, "models/tfidf.joblib")
print("Modèle sauvegardé dans models/tfidf.joblib")
