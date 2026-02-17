import streamlit as st
import pandas as pd

# Configuração da página
st.set_page_config(page_title="IMDB Recommender", layout="wide")

st.title("🎬 IMDB Top 100 - Recomendador Inteligente")

# Cache para não recarregar o CSV toda hora
@st.cache_data
def load_data():
    df = pd.read_csv("top100_clean.csv")
    df["genres"] = df["genres"].str.split(",")
    df = df.explode("genres")
    df["genres"] = df["genres"].str.strip()
    return df

df = load_data()

# Sidebar
st.sidebar.header("Filtros")

genres = sorted(df["genres"].unique())
selected_genre = st.sidebar.selectbox("Escolha um gênero", genres)

min_rating = st.sidebar.slider(
    "Nota mínima",
    min_value=float(df["averageRating"].min()),
    max_value=float(df["averageRating"].max()),
    value=8.0
)

# Filtro principal
filtered = df[
    (df["genres"] == selected_genre) &
    (df["averageRating"] >= min_rating)
]

st.subheader(f"🎥 Filmes de {selected_genre} com nota acima de {min_rating}")

if filtered.empty:
    st.warning("Nenhum filme encontrado com esses critérios.")
else:
    top_movies = filtered.sort_values(
        by="averageRating", ascending=False
    ).head(10)

    st.dataframe(
        top_movies[["primaryTitle", "startYear", "averageRating"]],
        use_container_width=True
    )

st.markdown("---")
st.caption("Projeto IMDB Recommender 🚀")
