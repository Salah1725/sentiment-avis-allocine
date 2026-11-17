"""API de classification d'avis de films.

Lancer depuis le dossier avis-allocine :
    .venv\\Scripts\\uvicorn api:app --reload
Documentation interactive : http://127.0.0.1:8000/docs
"""
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import modeles

app = FastAPI(
    title="Sentiment des avis Allociné",
    description="Classe un avis de film en français (positif / négatif) avec "
                "TF-IDF, CamemBERT fine-tuné ou un LLM local (Qwen) en zero-shot.",
)


class Avis(BaseModel):
    texte: str = Field(min_length=3, max_length=20_000,
                       examples=["Film pas mauvais, qui se laisse regarder."])
    modele: Literal["tfidf", "camembert", "qwen"] = "camembert"


class Prediction(BaseModel):
    modele: str
    sentiment: Literal["positif", "negatif"]
    proba_positif: Optional[float] = Field(
        None, description="Probabilité que l'avis soit positif (absente pour un LLM)")
    secondes: float


@app.get("/health")
def health():
    """Indique quels modèles sont disponibles."""
    return {"status": "ok", "modeles_disponibles": modeles.disponibles()}


@app.post("/predict", response_model=Prediction)
def predict(avis: Avis):
    try:
        return modeles.predire(avis.texte, avis.modele)
    except modeles.ModeleIndisponible as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
