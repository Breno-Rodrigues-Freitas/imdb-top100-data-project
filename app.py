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

query = "SELECT title, year, rating, votes FROM movies WHERE rating >= ?"
params = [min_rating]

if selected_genre != "All":
    query += " AND genre = ?"
    params.append(selected_genre)

if search_title:
    query += " AND title LIKE ?"
    params.append(f"%{search_title}%")

query += " ORDER BY rating DESC"

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
    else:
        st.error("Movie not found online either.")
else:
    st.dataframe(df, use_container_width=True)

# ==============================
# DASHBOARD SECTION
# ==============================

st.markdown("---")
st.subheader("📊 Movies Distribution by Genre")

genre_count_query = """
SELECT genre, COUNT(*) as total_movies
FROM movies
GROUP BY genre
ORDER BY total_movies DESC
"""

genre_df = pd.read_sql(genre_count_query, conn)

st.bar_chart(genre_df.set_index("genre"))

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
