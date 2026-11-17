"""Démo : on tape un avis de film et on compare les réponses des modèles.

Lancer depuis le dossier avis-allocine :
    .venv\\Scripts\\streamlit run app.py
"""
from pathlib import Path

import pandas as pd
import streamlit as st

import modeles

st.set_page_config(page_title="Sentiment des avis de films", page_icon="🎬", layout="wide")

EXEMPLES = {
    "« Pas mauvais » (négation)": "Film pas mauvais, qui se laisse regarder. Très belle performance "
                                  "de Wahlberg pas habitué à jouer ce genre de rôle et qui réussit son pari.",
    "« Génial »… mais pour un autre film": "Après avoir réussi à mêler horreur et teen-movie avec le génial "
                                          "Destination finale, le réalisateur s'est embarqué dans un film "
                                          "de science-fiction. Verdict : c'est sympa, mais c'est hélas très "
                                          "lourd, répétitif et peu captivant.",
    "Ironie": "Ouééé tro bi1 ce film, c un chef d'oeuvre comme vous dites ici lol ^^",
}


@st.cache_resource
def modeles_disponibles():
    return modeles.disponibles()


st.title("🎬 Sentiment des avis de films en français")
st.write("Trois approches comparées sur 200 000 avis Allociné : **TF-IDF** (classique), "
         "**CamemBERT** fine-tuné (Transformer) et **Qwen 2.5 3B** en zero-shot (LLM local, sans entraînement).")

dispo = modeles_disponibles()
if not dispo:
    st.error("Aucun modèle disponible : lance `entrainer_tfidf.py` et/ou télécharge CamemBERT.")
    st.stop()

choix_exemple = st.selectbox("Choisir un exemple (ou écrire ton propre avis)", ["—"] + list(EXEMPLES))
texte = st.text_area("Avis", value=EXEMPLES.get(choix_exemple, ""), height=150)
choisis = st.multiselect("Modèles", dispo, default=dispo, format_func=modeles.NOMS.get)

if st.button("Analyser", type="primary", disabled=len(texte.strip()) < 3 or not choisis):
    colonnes = st.columns(len(choisis))
    for colonne, nom in zip(colonnes, choisis):
        with colonne:
            st.subheader(modeles.NOMS[nom])
            try:
                with st.spinner("Analyse…"):
                    r = modeles.predire(texte, nom)
            except modeles.ModeleIndisponible as e:
                st.warning(str(e))
                continue
            emoji = "👍" if r["sentiment"] == "positif" else "👎"
            st.metric("Sentiment", f"{emoji} {r['sentiment']}")
            if r["proba_positif"] is not None:
                st.progress(r["proba_positif"], text=f"Probabilité positif : {r['proba_positif']:.1%}")
            else:
                st.caption("Un LLM donne une réponse, pas une probabilité.")
            st.caption(f"Temps de réponse : {1000 * r['secondes']:.0f} ms")

st.divider()
st.subheader("Résultats sur le jeu de validation")
st.dataframe(pd.DataFrame({
    "Modèle": ["TF-IDF + bigrammes", "CamemBERT fine-tuné", "Qwen 2.5 3B (zero-shot)"],
    "Accuracy": ["93,8 %", "95,8 %", "95,4 % (± 1,8, 500 avis)"],
    "Avis d'entraînement": ["160 000", "20 000", "0"],
}), hide_index=True)

if Path("resultats/vitesse.csv").exists():
    st.caption("Vitesse mesurée sur mon PC (CPU), mêmes 500 avis :")
    st.dataframe(pd.read_csv("resultats/vitesse.csv"), hide_index=True)
