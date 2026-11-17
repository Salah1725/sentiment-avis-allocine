"""Tests de l'API. Lancer : .venv\\Scripts\\python -m pytest -q"""
import pytest
from fastapi.testclient import TestClient

import modeles
from api import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert "tfidf" in r.json()["modeles_disponibles"]


@pytest.mark.parametrize("texte, attendu", [
    ("Un chef-d'œuvre, magnifique et bouleversant. À voir absolument !", "positif"),
    ("Un navet ennuyeux et ridicule, à éviter.", "negatif"),
])
def test_predict_tfidf(texte, attendu):
    r = client.post("/predict", json={"texte": texte, "modele": "tfidf"})
    assert r.status_code == 200
    corps = r.json()
    assert corps["sentiment"] == attendu
    assert 0 <= corps["proba_positif"] <= 1


def test_texte_trop_court():
    r = client.post("/predict", json={"texte": "ok", "modele": "tfidf"})
    assert r.status_code == 422


def test_modele_inconnu():
    r = client.post("/predict", json={"texte": "Super film", "modele": "gpt"})
    assert r.status_code == 422


def test_modele_indisponible(monkeypatch):
    def absent(*args):
        raise modeles.ModeleIndisponible("absent")

    monkeypatch.setattr(modeles, "proba_camembert", absent)
    r = client.post("/predict", json={"texte": "Super film", "modele": "camembert"})
    assert r.status_code == 503
