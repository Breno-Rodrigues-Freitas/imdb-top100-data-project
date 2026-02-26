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
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

# ==============================
# CONFIG
# ==============================

st.set_page_config(page_title="IMDB Movie Recommender", layout="wide")
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

def fetch_trailer(title):
    if not YOUTUBE_API_KEY:
        return None

    search_url = "https://www.googleapis.com/youtube/v3/search"

    params = {
        "part": "snippet",
        "q": f"{title} official trailer",
        "key": YOUTUBE_API_KEY,
        "type": "video",
        "maxResults": 1
    }

    response = requests.get(search_url, params=params)
    data = response.json()

    if "items" in data and len(data["items"]) > 0:
        return data["items"][0]["id"]["videoId"]

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

        video_id = fetch_trailer(api_movie["Title"])

        if video_id:
            st.video(f"https://www.youtube.com/watch?v={video_id}")
        else:
            st.warning("Trailer not found.")

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
# ENHANCED CHATBOT - SOMENTE ISSO FOI MELHORADO
# ==============================

st.markdown("---")
st.subheader("🤖 Movie Assistant")

# Initialize chat session
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = [
        {"role": "assistant", "content": "👋 Hi! I can help you find movies. Try asking:"},
        {"role": "assistant", "content": "• 'Show action movies'\n• 'Top 5 movies'\n• 'Movies like Inception'\n• 'Recommend something'"}
    ]

# Display chat messages
for message in st.session_state.chat_messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
if user_input := st.chat_input("Ask about movies..."):

    # Add user message
    st.session_state.chat_messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # Process user input
    response = ""
    user_input_lower = user_input.lower()

    # Check for genre
    genres_list = ['action', 'comedy', 'drama', 'horror', 'romance', 'sci-fi', 'thriller']
    found_genre = None
    for genre in genres_list:
        if genre in user_input_lower:
            found_genre = genre
            break

    if found_genre:
        # Query movies by genre
        genre_query = """
        SELECT m.title, m.rating, m.year
        FROM movies m
        JOIN movie_genres mg ON m.id = mg.movie_id
        JOIN genres g ON g.id = mg.genre_id
        WHERE LOWER(g.name) LIKE ? AND m.rating >= 6.0
        ORDER BY m.rating DESC
        LIMIT 5
        """
        genre_df = pd.read_sql(genre_query, conn, params=[f"%{found_genre}%"])
        
        if not genre_df.empty:
            response = f"🎬 **Top {found_genre} movies:**\n\n"
            for i, row in genre_df.iterrows():
                response += f"{i+1}. **{row['title']}** ({row['year']}) - ⭐ {row['rating']:.1f}\n"
        else:
            response = f"Sorry, no {found_genre} movies found in database."

    elif "top" in user_input_lower or "best" in user_input_lower:
        # Extract number if present
        import re
        numbers = re.findall(r'\d+', user_input)
        limit = int(numbers[0]) if numbers else 5
        
        top_query = """
        SELECT title, rating, year
        FROM movies
        WHERE rating IS NOT NULL
        ORDER BY rating DESC
        LIMIT ?
        """
        top_df = pd.read_sql(top_query, conn, params=[limit])
        
        if not top_df.empty:
            response = f"🏆 **Top {limit} Movies:**\n\n"
            for i, row in top_df.iterrows():
                response += f"{i+1}. **{row['title']}** ({row['year']}) - ⭐ {row['rating']:.1f}\n"
        else:
            response = "No movies found in database."

    elif "like" in user_input_lower or "similar" in user_input_lower:
        # Extract movie title
        words = user_input.split()
        if "like" in words:
            idx = words.index("like")
            if idx + 1 < len(words):
                movie_title = " ".join(words[idx+1:]).strip('?.,!').title()
                
                # Get similar movies (using existing similarity logic)
                if not movies_df.empty:
                    if movie_title in movies_df["title"].values:
                        idx = movies_df[movies_df["title"] == movie_title].index[0]
                        sim_scores = list(enumerate(cosine_sim[idx]))
                        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:4]
                        similar = movies_df["title"].iloc[[i[0] for i in sim_scores]].tolist()
                        
                        response = f"🎯 **Movies like '{movie_title}':**\n\n"
                        for i, movie in enumerate(similar, 1):
                            response += f"{i}. {movie}\n"
                    else:
                        response = f"Movie '{movie_title}' not found in database."
                else:
                    response = "No movies in database for comparison."

    elif "recommend" in user_input_lower or "suggest" in user_input_lower:
        # Random recommendations
        rec_query = """
        SELECT title, rating, year
        FROM movies
        WHERE rating >= 7.0
        ORDER BY RANDOM()
        LIMIT 5
        """
        rec_df = pd.read_sql(rec_query, conn)
        
        if not rec_df.empty:
            response = "🎲 **Random Recommendations:**\n\n"
            for i, row in rec_df.iterrows():
                response += f"{i+1}. **{row['title']}** ({row['year']}) - ⭐ {row['rating']:.1f}\n"
        else:
            response = "No movies available for recommendations."

    elif "trailer" in user_input_lower:
        # Extract movie title for trailer
        words = user_input.split()
        if "for" in words:
            idx = words.index("for")
            if idx + 1 < len(words):
                movie_title = " ".join(words[idx+1:]).strip('?.,!')
                video_id = fetch_trailer(movie_title)
                if video_id:
                    response = f"TRAILER:{movie_title}:{video_id}"
                else:
                    response = f"Sorry, couldn't find trailer for '{movie_title}'."
        else:
            response = "Which movie? Try: 'trailer for Inception'"

    elif "help" in user_input_lower:
        response = """**Available commands:**
• 'Show action movies' - movies by genre
• 'Top 10 movies' - best rated
• 'Movies like Inception' - similar movies
• 'Recommend something' - random picks
• 'Trailer for Inception' - watch trailer"""

    else:
        response = "I didn't understand. Try 'help' to see available commands."

    # Add assistant response
    st.session_state.chat_messages.append({"role": "assistant", "content": response})
    
    # Display response
    with st.chat_message("assistant"):
        if response.startswith("TRAILER:"):
            parts = response.split(":")
            if len(parts) == 3:
                _, movie_title, video_id = parts
                st.write(f"🎬 **Trailer for {movie_title}:**")
                st.video(f"https://www.youtube.com/watch?v={video_id}")
            else:
                st.write(response)
        else:
            st.write(response)
