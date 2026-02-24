import streamlit as st
import pandas as pd
import sqlite3
import requests
import os
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==============================
# LOAD ENV
# ==============================

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
# CREATE USERS + HISTORY TABLES
# ==============================

conn.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE,
    password TEXT
)
""")

conn.execute("""
CREATE TABLE IF NOT EXISTS history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    movie_title TEXT,
    searched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(user_id) REFERENCES users(id)
)
""")

conn.commit()

# ==============================
# AUTH SYSTEM
# ==============================

if "user_id" not in st.session_state:
    st.session_state.user_id = None

if st.session_state.user_id is None:

    st.subheader("🔐 Login or Register")

    auth_option = st.radio("Choose option", ["Login", "Register"])

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if auth_option == "Register":
        if st.button("Create Account"):
            try:
                conn.execute(
                    "INSERT INTO users (username, password) VALUES (?, ?)",
                    (username, password)
                )
                conn.commit()
                st.success("Account created! Please login.")
            except:
                st.error("Username already exists.")

    if auth_option == "Login":
        if st.button("Login"):
            user = conn.execute(
                "SELECT id FROM users WHERE username = ? AND password = ?",
                (username, password)
            ).fetchone()

            if user:
                st.session_state.user_id = user[0]
                st.rerun()
            else:
                st.error("Invalid credentials.")

    st.stop()

# Logout
if st.button("Logout"):
    st.session_state.user_id = None
    st.rerun()

# ==============================
# OMDB FETCH FUNCTION
# ==============================

def fetch_movie_from_api(title):
    if not OMDB_API_KEY:
        st.error("OMDB_API_KEY not found.")
        return None

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

try:
    genres_query = "SELECT DISTINCT genre FROM movies ORDER BY genre"
    genres = pd.read_sql(genres_query, conn)["genre"].tolist()
except:
    genres = []

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

if selected_genre != "All":
    query += " AND m.id IN (SELECT movie_id FROM movie_genres mg JOIN genres g ON g.id = mg.genre_id WHERE g.name = ?)"
    params.append(selected_genre)

if search_title:
    query += " AND m.title LIKE ?"
    params.append(f"%{search_title}%")

query += " GROUP BY m.id ORDER BY m.rating DESC"

df = pd.read_sql(query, conn, params=params)

# ==============================
# RESULTS SECTION
# ==============================

st.subheader("🎥 Results")

if df.empty and search_title:

    st.info("Movie not found in local database. Searching online... 🌐")

    api_movie = fetch_movie_from_api(search_title)

    if api_movie:

        # salvar histórico
        conn.execute(
            "INSERT INTO history (user_id, movie_title) VALUES (?, ?)",
            (st.session_state.user_id, search_title)
        )
        conn.commit()

        st.subheader("🌍 Online Result")
        st.write(f"**Title:** {api_movie['Title']}")
        st.write(f"**Year:** {api_movie['Year']}")
        st.write(f"**Genre:** {api_movie['Genre']}")
        st.write(f"**IMDB Rating:** {api_movie['Rating']}")
        st.write(f"**Votes:** {api_movie['Votes']}")
        st.write(f"**Plot:** {api_movie['Plot']}")

    st.subheader("🎬 Trailer")

    trailer_query = api_movie["Title"] + " official trailer"
    youtube_search_url = f"https://www.youtube.com/results?search_query={trailer_query.replace(' ', '+')}"

    st.markdown(f"[▶ Watch Trailer on YouTube]({youtube_search_url})")

    if st.button("➕ Add to Local Database"):

            cursor = conn.cursor()

            cursor.execute("SELECT id FROM movies WHERE title = ?", (api_movie["Title"],))
            existing = cursor.fetchone()

            if not existing:

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
                    cursor.execute("INSERT OR IGNORE INTO genres (name) VALUES (?)", (genre,))
                    cursor.execute("SELECT id FROM genres WHERE name = ?", (genre,))
                    genre_id = cursor.fetchone()[0]
                    cursor.execute(
                        "INSERT INTO movie_genres (movie_id, genre_id) VALUES (?, ?)",
                        (movie_id, genre_id)
                    )

                conn.commit()
                st.success("Movie successfully added! 🚀")

    else:
        st.error("Movie not found online either.")

elif not df.empty:

    # salvar histórico
    if search_title:
        conn.execute(
            "INSERT INTO history (user_id, movie_title) VALUES (?, ?)",
            (st.session_state.user_id, search_title)
        )
        conn.commit()

    st.dataframe(df, use_container_width=True)

# ==============================
# USER HISTORY
# ==============================

st.markdown("---")
st.subheader("📜 Your Search History")

history_df = pd.read_sql(
    """
    SELECT movie_title, searched_at
    FROM history
    WHERE user_id = ?
    ORDER BY searched_at DESC
    LIMIT 10
    """,
    conn,
    params=(st.session_state.user_id,)
)

if not history_df.empty:
    st.dataframe(history_df, use_container_width=True)
else:
    st.info("No searches yet.")

# ==============================
# TOP 5
# ==============================

st.markdown("---")
st.subheader("🏆 Top 5 Movies")

top5_df = pd.read_sql("""
SELECT title, year, rating, votes
FROM movies
ORDER BY rating DESC, votes DESC
LIMIT 5
""", conn)

st.dataframe(top5_df, use_container_width=True)

# ==============================
# SIMILARITY
# ==============================

st.markdown("---")
st.subheader("🎯 Find Similar Movies")

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

    movies_df["content"] = movies_df["genres"]

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(movies_df["content"])
    cosine_sim = cosine_similarity(tfidf_matrix)

    selected_movie = st.selectbox("Choose a movie", movies_df["title"])

    if selected_movie:
        idx = movies_df[movies_df["title"] == selected_movie].index[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
        similar_indices = [i[0] for i in sim_scores]

        st.write("### 🔥 Similar Movies:")
        st.write(movies_df["title"].iloc[similar_indices].values)

# ==============================
# CHATBOT
# ==============================

st.markdown("---")
st.subheader("🤖 Movie Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

user_input = st.chat_input("Ask for movie recommendations...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.write(user_input)

    if "action" in user_input.lower():
        action_df = pd.read_sql(
            "SELECT title FROM movies WHERE rating >= 7 LIMIT 5",
            conn
        )
        response = "Here are some action movies:\n"
        response += "\n".join(action_df["title"].tolist())
    else:
        response = "Tell me a genre like action, drama, comedy..."

    st.session_state.messages.append({"role": "assistant", "content": response})

    with st.chat_message("assistant"):
        st.write(response)
