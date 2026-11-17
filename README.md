# Sentiment des avis de films en français : TF-IDF vs CamemBERT vs LLM

Classification d'avis Allociné (positif / négatif) avec trois approches très différentes, comparées sur **la précision, la vitesse, le coût et les données nécessaires** :

1. **TF-IDF + régression logistique** : l'approche classique, entraînée de zéro ;
2. **CamemBERT fine-tuné** : un Transformer pré-entraîné sur du français, ajusté sur nos avis ;
3. **un LLM en zero-shot** (Qwen 2.5 3B en local via Ollama) : aucun entraînement, juste une consigne.

Le projet va jusqu'au produit : modèles sauvegardés, **API FastAPI** testée et **démo Streamlit**.

> Projet personnel d'autoformation en NLP.
> Données : [Allociné](https://huggingface.co/datasets/tblard/allocine) (200 000 avis, licence MIT) : 160 000 pour l'entraînement, 20 000 pour la validation, 20 000 pour le test.

## Résultats

| | TF-IDF + bigrammes | CamemBERT fine-tuné | Qwen 2.5 3B zero-shot |
|---|---|---|---|
| **Accuracy (validation, 20 000 avis)** | 93,8 % | **95,8 %** | — |
| **Accuracy (mêmes 500 avis)** | 93,2 % | 95,2 % | **95,6 %** |
| Avis d'entraînement | 160 000 | 20 000 | **0** |
| Entraînement | quelques minutes (CPU) | 4 min (GPU T4, Colab) | aucun |
| Latence, un avis (CPU) | **0,6 ms** | 83 ms | 285 ms |
| Débit, par paquets (CPU) | **~6 200 avis/s** | ~6 avis/s | ~3,4 avis/s |
| Taille du modèle | 10 Mo | 440 Mo | 1,9 Go |
| Avis pièges réussis (voir plus bas) | 1 / 3 | 3 / 3 | 3 / 3 |

Mesures de vitesse : [`mesurer_vitesse.py`](mesurer_vitesse.py), sur mon PC, **CPU uniquement** (résultats dans [`resultats/vitesse.csv`](resultats/vitesse.csv)). Sur 500 avis, la marge d'erreur de l'accuracy est d'environ ± 1,8 point : CamemBERT et Qwen ne sont pas distinguables.

### Ce que j'en retiens
- **CamemBERT fait 32 % d'erreurs en moins que TF-IDF** (4,2 % contre 6,2 % sur 20 000 avis) avec **8 fois moins de données**, grâce au pré-entraînement.
- **Un petit LLM local, sans aucun exemple étiqueté, fait aussi bien que CamemBERT** : sur les mêmes 500 avis, l'écart est dans la marge d'erreur. Il fait mieux que TF-IDF entraîné sur 160 000 avis.
- **La précision se paie en vitesse** : sur CPU, CamemBERT est environ 140 fois plus lent que TF-IDF, et Qwen environ 450 fois. Sur CPU, traiter CamemBERT par paquets n'aide même pas : les avis courts sont complétés (*padding*) jusqu'à la longueur du plus long, ce qui gaspille du calcul. Un GPU changerait complètement ces chiffres.
- **Pour classer un million d'avis par jour**, je choisirais CamemBERT sur GPU, ou TF-IDF si le budget est minimal et que 2 points d'accuracy en moins sont acceptables. **Pour un prototype sans données étiquetées**, un LLM en zero-shot.
- **Les deux se combinent** : un LLM peut étiqueter des données, sur lesquelles on entraîne ensuite un petit modèle rapide pour la production (une forme de distillation).

## Analyse d'erreurs

Les chiffres ne suffisent pas : j'ai lu les erreurs des modèles.

- **Corrélation trompeuse** : parmi les mots les plus « négatifs » pour TF-IDF figure `plv`, la **signature d'un auteur** d'avis, pas un mot de critique. Le modèle a appris un raccourci propre au jeu de données (*shortcut learning*).
- **Négation** : « pas mauvais » est lu comme négatif par TF-IDF, qui additionne des poids de mots sans regarder leur ordre. Les bigrammes n'aident qu'à moitié : dans les données, « pas mauvais » apparaît surtout dans des avis… négatifs (88 %), par exemple « pas mauvais, mais… ».
- **Troncature** : CamemBERT ne lit que les 256 premiers tokens ; 15 % des avis sont coupés. Risque : un verdict placé à la fin de l'avis disparaît.
- **Les erreurs du LLM** sont des cas difficiles (ironie, verdict à la fin après une liste de défauts, critique hors sujet), et **une partie vient des étiquettes** : un avis au texte clairement positif peut être noté négatif. Ce bruit fixe un plafond qu'aucun modèle ne dépassera.

### Les trois avis pièges

| Avis | Vraie réponse | TF-IDF | CamemBERT | Qwen |
|---|---|---|---|---|
| Critique qui cite un **autre** film « génial » | négatif | ✅ | ✅ | ✅ |
| « Film **pas mauvais**, qui se laisse regarder… » | positif | ❌ | ✅ | ✅ |
| Avis inventé : « Un film d'un grand intérêt, pas mauvais du tout. » | positif | ❌ | ✅ | ✅ |

## Structure du projet

```
telecharger_donnees.py  télécharge le jeu Allociné (data/*.parquet)
1_exploration.ipynb     exploration des données (longueur des avis, avis pièges)
2_tfidf.ipynb           TF-IDF + régression logistique, coefficients, bigrammes, analyse d'erreurs
3_camembert.ipynb       tokenisation en sous-mots, fine-tuning de CamemBERT (Google Colab, GPU)
4_llm.ipynb             LLM en zero-shot (Qwen via Ollama ; Claude via l'API Anthropic en option)
entrainer_tfidf.py      entraîne et sauvegarde le pipeline TF-IDF (models/tfidf.joblib)
modeles.py              chargement des 3 modèles et prédiction (code partagé)
mesurer_vitesse.py      latence et débit des modèles sur les mêmes 500 avis
api.py                  API FastAPI (/predict, /health)
test_api.py             tests de l'API (pytest)
app.py                  démo Streamlit
```

## Lancer le projet

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt

python telecharger_donnees.py   # télécharge Allociné dans data/
python entrainer_tfidf.py       # entraîne TF-IDF → models/tfidf.joblib
```

- **CamemBERT** : placer le modèle fine-tuné dans `models/camembert-allocine/` (produit par `3_camembert.ipynb`), ou indiquer un identifiant Hugging Face avec la variable d'environnement `CAMEMBERT_PATH`.
- **Qwen** : installer [Ollama](https://ollama.com), puis `ollama pull qwen2.5:3b`.

Chaque modèle est optionnel : l'API et la démo utilisent ceux qui sont disponibles.

```bash
uvicorn api:app --reload        # API : http://127.0.0.1:8000/docs
streamlit run app.py            # démo : http://localhost:8501
python -m pytest -q             # tests
```

Exemple d'appel :

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"texte\": \"Film pas mauvais, qui se laisse regarder.\", \"modele\": \"tfidf\"}"
```

## Limites
- L'accuracy des LLM est mesurée sur 500 avis seulement (marge d'erreur ± 1,8 point), pour limiter le temps de calcul.
- CamemBERT n'a été entraîné que sur 20 000 avis et une époque, sur un GPU gratuit ; plus de données ou d'époques amélioreraient sans doute le score.
- Les étiquettes Allociné viennent des notes des spectateurs : une partie est bruitée.
- Le jeu de **test** n'a pas encore été utilisé : tous les choix ont été faits sur la validation, et il servira pour la mesure finale.
