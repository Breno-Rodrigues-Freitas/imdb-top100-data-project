import streamlit as st
import pandas as pd
import sqlite3
import requests
import os
from dotenv import load_dotenv

load_dotenv()

OMDB_API_KEY = os.getenv("OMDB_API_KEY")

# ==============================
# CONFIG
# ==============================

st.set_page_config(page_title="IMDB Recommender", layout="wide")
st.title("🎬 IMDB Movie Recommender - Hybrid Version")

# ==============================
# DATABASE CONNECTION
# ==============================

@st.cache_resource
def get_connection():
    return sqlite3.connect("movies.db", check_same_thread=False)

conn = get_connection()

# ==============================
# OMDB FETCH FUNCTION
# ==============================

def fetch_movie_from_api(title):
    url = f"http://www.omdbapi.com/?t={title}&apikey={OMDB_API_KEY}"
    response = requests.get(url)
    data = response.json()

    if data.get("Response") == "True":
        return {
            "Title": data.get("Title"),
            "Year": data.get("Year"),
            "Genre": data.get("Genre"),
            "Rating": data.get("imdbRating"),
            "Votes": data.get("imdbVotes"),
            "Plot": data.get("Plot")
        }
    return None

# ==============================
# SIDEBAR FILTERS
# ==============================

st.sidebar.header("Filters")

genres_query = "SELECT DISTINCT genre FROM movies ORDER BY genre"
genres = pd.read_sql(genres_query, conn)["genre"].tolist()

search_title = st.sidebar.text_input("Search movie title")
selected_genre = st.sidebar.selectbox("Select genre", ["All"] + genres)
min_rating = st.sidebar.slider("Minimum rating", 0.0, 10.0, 8.0)

# ==============================
# LOCAL DATABASE QUERY
# ==============================

query = """
SELECT 
    m.title,
    m.year,
    m.rating,
    m.votes,
    GROUP_CONCAT(g.name, ', ') as genres
FROM movies m
JOIN movie_genres mg ON m.id = mg.movie_id
JOIN genres g ON g.id = mg.genre_id
WHERE m.rating >= ?
"""

params = [min_rating]

# Filtro por gênero
if selected_genre != "All":
    query += " AND m.id IN (SELECT movie_id FROM movie_genres mg JOIN genres g ON g.id = mg.genre_id WHERE g.name = ?)"
    params.append(selected_genre)

# Filtro por título
if search_title:
    query += " AND m.title LIKE ?"
    params.append(f"%{search_title}%")

query += """
GROUP BY m.id
ORDER BY m.rating DESC
"""

df = pd.read_sql(query, conn, params=params)

# ==============================
# RESULTS SECTION
# ==============================

st.subheader("🎥 Results")

if df.empty and search_title:
    st.info("Movie not found in local database. Searching online... 🌐")

    api_movie = fetch_movie_from_api(search_title)

    if api_movie:
        st.subheader("🌍 Online Result")
        st.write(f"**Title:** {api_movie['Title']}")
        st.write(f"**Year:** {api_movie['Year']}")
        st.write(f"**Genre:** {api_movie['Genre']}")
        st.write(f"**IMDB Rating:** {api_movie['Rating']}")
        st.write(f"**Votes:** {api_movie['Votes']}")
        st.write(f"**Plot:** {api_movie['Plot']}")

        if st.button("➕ Add to Local Database"):

            cursor = conn.cursor()

            cursor.execute("SELECT id FROM movies WHERE title = ?", (api_movie["Title"],))
            existing = cursor.fetchone()

            if existing:
                st.warning("Movie already exists in database.")
            else:
                year = int(api_movie["Year"]) if str(api_movie["Year"]).isdigit() else None
                rating = float(api_movie["Rating"]) if api_movie["Rating"] != "N/A" else 0
                votes = int(api_movie["Votes"].replace(",", "")) if api_movie["Votes"] != "N/A" else 0

                cursor.execute(
                    "INSERT INTO movies (title, year, rating, votes) VALUES (?, ?, ?, ?)",
                    (api_movie["Title"], year, rating, votes)
                )

                movie_id = cursor.lastrowid

                genres_list = [g.strip() for g in api_movie["Genre"].split(",")]

                for genre in genres_list:
                    cursor.execute(
                        "INSERT OR IGNORE INTO genres (name) VALUES (?)",
                        (genre,)
                    )

                    cursor.execute(
                        "SELECT id FROM genres WHERE name = ?",
                        (genre,)
                    )
                    genre_id = cursor.fetchone()[0]

                    cursor.execute(
                        "INSERT INTO movie_genres (movie_id, genre_id) VALUES (?, ?)",
                        (movie_id, genre_id)
                    )

                conn.commit()
                st.success("Movie successfully added to local database! 🚀")

    else:
        st.error("Movie not found online either.")

# ==============================
# TOP 5 SECTION
# ==============================

st.markdown("---")
st.subheader("🏆 Top 5 Movies (Highest Rated)")

top5_query = """
SELECT title, year, rating, votes
FROM movies
ORDER BY rating DESC, votes DESC
LIMIT 5
"""

top5_df = pd.read_sql(top5_query, conn)

st.dataframe(top5_df, use_container_width=True)

st.markdown("---")
st.caption("IMDB Recommender - Hybrid SQLite + OMDb API 🚀")

# ==============================
# 🎯 SIMILARITY SECTION
# ==============================

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.markdown("---")
st.subheader("🎯 Find Similar Movies")

# Buscar todos os filmes com gêneros
similarity_query = """
SELECT 
    m.id,
    m.title,
    GROUP_CONCAT(g.name, ' ') as genres
FROM movies m
JOIN movie_genres mg ON m.id = mg.movie_id
JOIN genres g ON g.id = mg.genre_id
GROUP BY m.id
"""

movies_df = pd.read_sql(similarity_query, conn)

if not movies_df.empty:

    # Criar coluna texto
    movies_df["content"] = movies_df["genres"]

    # Criar matriz TF-IDF
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(movies_df["content"])

    # Similaridade
    cosine_sim = cosine_similarity(tfidf_matrix)

    # Selecionar filme
    selected_movie = st.selectbox("Choose a movie", movies_df["title"])

    if selected_movie:
        idx = movies_df[movies_df["title"] == selected_movie].index[0]

        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        sim_scores = sim_scores[1:6]  # top 5 similares

        similar_indices = [i[0] for i in sim_scores]

        st.write("### 🔥 Similar Movies:")
        st.write(movies_df["title"].iloc[similar_indices].values)
