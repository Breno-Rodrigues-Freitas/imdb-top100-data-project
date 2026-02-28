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
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

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

# Add profile columns to users table (if they don't exist)
try:
    conn.execute("ALTER TABLE users ADD COLUMN profile_pic TEXT")
except:
    pass  # Column already exists

try:
    conn.execute("ALTER TABLE users ADD COLUMN bio TEXT")
except:
    pass  # Column already exists

conn.commit()

# ==============================
# AUTH SYSTEM - IMPROVED
# ==============================

import hashlib
import hmac
import re
from datetime import datetime, timedelta

# Security constants
MIN_PASSWORD_LENGTH = 6
MAX_LOGIN_ATTEMPTS = 5
LOCKOUT_TIME = 15  # minutes

# Password hashing functions
def hash_password(password: str) -> str:
    """Create secure password hash using PBKDF2"""
    salt = os.urandom(32).hex()
    hash_obj = hashlib.pbkdf2_hmac(
        'sha256',
        password.encode('utf-8'),
        salt.encode('utf-8'),
        100000  # Number of iterations
    )
    return f"{salt}${hash_obj.hex()}"

def verify_password(password: str, hashed: str) -> bool:
    """Verify if password matches hash"""
    try:
        salt, hash_value = hashed.split('$')
        hash_obj = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000
        )
        return hmac.compare_digest(hash_obj.hex(), hash_value)
    except:
        return False

def validate_username(username: str) -> tuple[bool, str]:
    """Validate username format"""
    if not username or len(username) < 3:
        return False, "Username must have at least 3 characters"
    if len(username) > 30:
        return False, "Username must have at most 30 characters"
    if not re.match("^[a-zA-Z0-9_]+$", username):
        return False, "Username can only contain letters, numbers and underscore"
    return True, "Valid username"

def validate_password(password: str) -> tuple[bool, str]:
    """Validate password strength"""
    if len(password) < MIN_PASSWORD_LENGTH:
        return False, f"Password must have at least {MIN_PASSWORD_LENGTH} characters"
    if len(password) > 50:
        return False, "Password must have at most 50 characters"
    return True, "Valid password"

# Create login attempts table (if not exists)
conn.execute("""
CREATE TABLE IF NOT EXISTS login_attempts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT,
    ip_address TEXT,
    attempt_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    success BOOLEAN
)
""")

# Create indexes for better performance
conn.execute("CREATE INDEX IF NOT EXISTS idx_login_attempts_username ON login_attempts(username)")
conn.execute("CREATE INDEX IF NOT EXISTS idx_login_attempts_time ON login_attempts(attempt_time)")

conn.commit()

# Initialize session state
if "user_id" not in st.session_state:
    st.session_state.user_id = None
    st.session_state.username = None
    st.session_state.login_time = None
    st.session_state.remember_me = False

def check_login_attempts(username: str) -> tuple[bool, str]:
    """Check if user has exceeded login attempts"""
    try:
        # Count failed attempts in last X minutes
        cutoff = (datetime.now() - timedelta(minutes=LOCKOUT_TIME)).strftime("%Y-%m-%d %H:%M:%S")
        
        attempts = conn.execute("""
            SELECT COUNT(*) FROM login_attempts 
            WHERE username = ? AND success = 0 AND attempt_time > ?
        """, (username, cutoff)).fetchone()[0]
        
        if attempts >= MAX_LOGIN_ATTEMPTS:
            return False, f"Too many failed attempts. Please try again in {LOCKOUT_TIME} minutes."
        
        return True, ""
    except:
        return True, ""

def record_login_attempt(username: str, success: bool):
    """Record login attempt in database"""
    try:
        ip_address = "unknown"  # Streamlit doesn't expose real IP easily
        
        conn.execute(
            "INSERT INTO login_attempts (username, ip_address, success) VALUES (?, ?, ?)",
            (username, ip_address, success)
        )
        conn.commit()
    except:
        pass  # Silently fail if can't record

# Login/Register interface
if st.session_state.user_id is None:
    
    st.markdown("""
    <style>
    .auth-container {
        max-width: 400px;
        margin: 0 auto;
        padding: 20px;
        border-radius: 10px;
        background-color: #f8f9fa;
    }
    </style>
    """, unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="auth-container">', unsafe_allow_html=True)
        
        st.subheader("🔐 Welcome to Movie Recommender")
        
        tab1, tab2, tab3 = st.tabs(["🔑 Login", "📝 Register", "❓ Forgot Password"])
        
        # ===== LOGIN TAB =====
        with tab1:
            with st.form("login_form"):
                username = st.text_input("Username", placeholder="Enter your username")
                password = st.text_input("Password", type="password", placeholder="Enter your password")
                
                col1, col2 = st.columns([1, 1])
                with col1:
                    remember_me = st.checkbox("Remember me")
                with col2:
                    st.write("")  # Empty space for alignment
                
                submitted = st.form_submit_button("🚀 Login", use_container_width=True)
                
                if submitted:
                    if not username or not password:
                        st.error("Please fill in all fields")
                    else:
                        # Check login attempts
                        can_login, msg = check_login_attempts(username)
                        if not can_login:
                            st.error(msg)
                        else:
                            # Find user
                            user = conn.execute(
                                "SELECT id, username, password FROM users WHERE username = ?",
                                (username,)
                            ).fetchone()
                            
                            if user and verify_password(password, user[2]):
                                # Successful login
                                record_login_attempt(username, True)
                                
                                st.session_state.user_id = user[0]
                                st.session_state.username = user[1]
                                st.session_state.login_time = datetime.now()
                                st.session_state.remember_me = remember_me
                                
                                st.success(f"Welcome back, {username}! 🎉")
                                st.rerun()
                            else:
                                # Failed login
                                record_login_attempt(username, False)
                                st.error("Invalid username or password")
        
        # ===== REGISTER TAB =====
        with tab2:
            with st.form("register_form"):
                new_username = st.text_input("Choose username", placeholder="Min 3 characters, letters/numbers/_")
                new_password = st.text_input("Choose password", type="password", 
                                           placeholder=f"Min {MIN_PASSWORD_LENGTH} characters")
                confirm_password = st.text_input("Confirm password", type="password")
                
                # Password strength indicator
                if new_password:
                    strength = "Weak"
                    color = "red"
                    if len(new_password) >= 8 and any(c.isdigit() for c in new_password) and any(c.isupper() for c in new_password):
                        strength = "Strong"
                        color = "green"
                    elif len(new_password) >= 6:
                        strength = "Medium"
                        color = "orange"
                    
                    st.markdown(f"Password strength: <span style='color:{color}'>{strength}</span>", 
                              unsafe_allow_html=True)
                
                submitted = st.form_submit_button("✨ Create Account", use_container_width=True)
                
                if submitted:
                    # Validate fields
                    valid_username, username_msg = validate_username(new_username)
                    valid_password, password_msg = validate_password(new_password)
                    
                    if not valid_username:
                        st.error(username_msg)
                    elif not valid_password:
                        st.error(password_msg)
                    elif new_password != confirm_password:
                        st.error("Passwords do not match")
                    else:
                        try:
                            # Check if user already exists
                            existing = conn.execute(
                                "SELECT id FROM users WHERE username = ?", 
                                (new_username,)
                            ).fetchone()
                            
                            if existing:
                                st.error("Username already exists")
                            else:
                                # Create new user
                                hashed = hash_password(new_password)
                                conn.execute(
                                    "INSERT INTO users (username, password) VALUES (?, ?)",
                                    (new_username, hashed)
                                )
                                conn.commit()
                                
                                st.success("✅ Account created successfully! Please login.")
                                
                        except Exception as e:
                            st.error(f"Error creating account: {str(e)}")
        
        # ===== FORGOT PASSWORD TAB =====
        with tab3:
            st.info("🔑 Password Reset")
            st.write("Enter your username and we'll help you reset your password.")
            
            with st.form("forgot_form"):
                reset_username = st.text_input("Username", placeholder="Enter your username")
                submitted = st.form_submit_button("🔄 Reset Password", use_container_width=True)
                
                if submitted:
                    if not reset_username:
                        st.error("Please enter your username")
                    else:
                        # Check if user exists
                        user = conn.execute(
                            "SELECT id FROM users WHERE username = ?", 
                            (reset_username,)
                        ).fetchone()
                        
                        if user:
                            # Here you would implement email sending
                            # For now, just a message
                            st.success("📧 Password reset instructions would be sent to your email.")
                            st.info("(Email functionality not implemented in this demo)")
                        else:
                            # Same message to not reveal if user exists
                            st.success("📧 If the username exists, instructions will be sent.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.stop()

# ==============================
# LOGOUT BUTTON (Improved)
# ==============================

with st.sidebar:
    # User profile card
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    ">
        <div style="display: flex; align-items: center; gap: 15px;">
            <div style="
                background: white;
                width: 50px;
                height: 50px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 24px;
            ">
                {st.session_state.username[0].upper() if st.session_state.username else '👤'}
            </div>
            <div style="color: white;">
                <div style="font-size: 18px; font-weight: 600;">{st.session_state.username}</div>
                <div style="font-size: 12px; opacity: 0.9;">
                    Logged in • {st.session_state.login_time.strftime("%H:%M") if st.session_state.login_time else "now"}
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Logout button
    if st.button("🚪 Logout", use_container_width=True):
        # Clear session
        for key in ['user_id', 'username', 'login_time', 'chat_messages']:
            if key in st.session_state:
                st.session_state[key] = None if key != 'chat_messages' else []
        st.rerun()
    
    st.markdown("---")
    st.header("🔍 Filters")

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
# YOUTUBE TRAILER FUNCTION
# ==============================

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
# HUGGING FACE AI FUNCTIONS
# ==============================

def get_movie_context():
    """Get movie data from database for AI context"""
    try:
        query = """
        SELECT title, rating, year 
        FROM movies 
        WHERE rating IS NOT NULL 
        ORDER BY rating DESC 
        LIMIT 20
        """
        df = pd.read_sql(query, conn)
        if not df.empty:
            return df.to_string(index=False)
        return "No movies in database yet."
    except:
        return "Database not available."

def get_ai_response(user_message):
    """Get response from Hugging Face AI model"""
    
    if not HUGGINGFACE_API_KEY:
        return "AI service not configured. Please check your HUGGINGFACE_API_KEY."
    
    try:
        # API endpoint
        API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"
        headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
        
        # Get movie context
        movie_context = get_movie_context()
        
        # Prepare prompt
        prompt = f"""<s>[INST] You are a movie assistant. Here are some movies in the database:
{movie_context}

Answer questions about movies. Keep answers short and helpful.

User: {user_message} [/INST]"""
        
        # Make request
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 150,
                "temperature": 0.7,
                "do_sample": True
            }
        }
        
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            
            # Extract text from response
            if isinstance(result, list):
                if len(result) > 0:
                    if isinstance(result[0], dict) and 'generated_text' in result[0]:
                        return result[0]['generated_text'].split('[/INST]')[-1].strip()
                    return str(result[0])
            elif isinstance(result, dict) and 'generated_text' in result:
                return result['generated_text'].split('[/INST]')[-1].strip()
            
            return "I received a response but couldn't understand it."
            
        elif response.status_code == 503:
            return "The AI model is loading. Please try again in 10 seconds."
        else:
            return f"Error {response.status_code}. Please try again."
            
    except requests.exceptions.Timeout:
        return "Request timed out. Please try again."
    except Exception as e:
        return f"Error: {str(e)}"

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
# PROFILE PAGE
# ==============================

def profile_page():
    st.subheader("👤 Your Profile")
    
    # Get user data
    user_data = conn.execute(
        "SELECT username, email, profile_pic, bio, favorite_genre, joined_date FROM users WHERE id = ?",
        (st.session_state.user_id,)
    ).fetchone()
    
    if user_data:
        username, email, profile_pic, bio, favorite_genre, joined_date = user_data
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### Profile Picture")
            
            # Display current profile picture
            if profile_pic and os.path.exists(profile_pic):
                st.image(profile_pic, width=200, caption="Current Photo")
            else:
                # Default avatar
                st.markdown(f"""
                <div style='
                    width: 200px;
                    height: 200px;
                    border-radius: 50%;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-size: 80px;
                    color: white;
                    margin-bottom: 20px;
                '>
                    {username[0].upper() if username else '👤'}
                </div>
                """, unsafe_allow_html=True)
            
            # Upload new picture
            uploaded_file = st.file_uploader(
                "Change profile picture", 
                type=['png', 'jpg', 'jpeg', 'gif'],
                key="profile_pic_uploader"
            )
            
            if uploaded_file is not None:
                # Show preview
                st.image(uploaded_file, width=150, caption="Preview")
                
                if st.button("💾 Save New Picture", use_container_width=True):
                    file_path = save_profile_picture(uploaded_file, username)
                    if file_path:
                        conn.execute(
                            "UPDATE users SET profile_pic = ? WHERE id = ?",
                            (file_path, st.session_state.user_id)
                        )
                        conn.commit()
                        st.success("Profile picture updated!")
                        st.rerun()
        
        with col2:
            st.markdown("### Profile Information")
            
            with st.form("profile_form"):
                # Username (read-only)
                st.text_input("Username", value=username, disabled=True)
                
                # Email
                new_email = st.text_input(
                    "Email", 
                    value=email if email else "",
                    placeholder="your@email.com"
                )
                
                # Bio
                new_bio = st.text_area(
                    "Bio", 
                    value=bio if bio else "",
                    placeholder="Tell us about your movie preferences...",
                    height=100
                )
                
                # Favorite Genre
                # Get all genres from database
                genres_df = pd.read_sql("SELECT name FROM genres ORDER BY name", conn)
                genre_list = ["None"] + genres_df["name"].tolist() if not genres_df.empty else ["None"]
                
                current_genre = favorite_genre if favorite_genre in genre_list else "None"
                new_genre = st.selectbox("Favorite Genre", genre_list, index=genre_list.index(current_genre) if current_genre in genre_list else 0)
                
                # Joined date (read-only)
                if joined_date:
                    st.text_input("Member Since", value=joined_date[:10], disabled=True)
                
                # Submit button
                submitted = st.form_submit_button("💾 Update Profile", use_container_width=True)
                
                if submitted:
                    # Update database
                    conn.execute(
                        """UPDATE users 
                           SET email = ?, bio = ?, favorite_genre = ? 
                           WHERE id = ?""",
                        (new_email, new_bio, new_genre if new_genre != "None" else None, st.session_state.user_id)
                    )
                    conn.commit()
                    st.success("Profile updated successfully!")
                    st.rerun()
    
    # Movie stats
    st.markdown("---")
    st.subheader("📊 Your Movie Stats")
    
    col1, col2, col3 = st.columns(3)
    
    # Total searches
    searches = conn.execute(
        "SELECT COUNT(*) FROM history WHERE user_id = ?",
        (st.session_state.user_id,)
    ).fetchone()[0]
    
    with col1:
        st.metric("🔍 Total Searches", searches)
    
    # Favorite genre (most searched)
    fav_genre = conn.execute("""
        SELECT 
            g.name,
            COUNT(*) as count
        FROM history h
        JOIN movies m ON m.title LIKE '%' || h.movie_title || '%'
        JOIN movie_genres mg ON m.id = mg.movie_id
        JOIN genres g ON g.id = mg.genre_id
        WHERE h.user_id = ?
        GROUP BY g.name
        ORDER BY count DESC
        LIMIT 1
    """, (st.session_state.user_id,)).fetchone()
    
    with col2:
        st.metric("🎯 Favorite Genre", fav_genre[0] if fav_genre else "N/A")
    
    # Movies added (if any)
    movies_added = conn.execute(
        "SELECT COUNT(*) FROM movies WHERE id IN (SELECT DISTINCT movie_id FROM movie_genres)"
    ).fetchone()[0]
    
    with col3:
        st.metric("📚 Total Movies", movies_added)

# ==============================
# UPDATE SIDEBAR WITH PROFILE LINK
# ==============================

with st.sidebar:
    # User profile card (improved)
    user_data = conn.execute(
        "SELECT profile_pic FROM users WHERE id = ?",
        (st.session_state.user_id,)
    ).fetchone()
    
    profile_pic = user_data[0] if user_data else None
    
    # Profile picture in card
    if profile_pic and os.path.exists(profile_pic):
        # Show image
        st.markdown(f"""
        <div style='
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            border-radius: 15px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        '>
            <div style="display: flex; align-items: center; gap: 15px;">
                <img src="data:image/png;base64,{get_profile_pic_base64(profile_pic)}" 
                     style="width: 50px; height: 50px; border-radius: 50%; object-fit: cover; border: 2px solid white;">
                <div style="color: white;">
                    <div style="font-size: 18px; font-weight: 600;">{st.session_state.username}</div>
                    <div style="font-size: 12px; opacity: 0.9;">
                        🕐 {st.session_state.login_time.strftime("%H:%M") if st.session_state.login_time else "now"}
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Default avatar
        st.markdown(f"""
        <div style='
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            border-radius: 15px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        '>
            <div style="display: flex; align-items: center; gap: 15px;">
                <div style='
                    background: white;
                    width: 50px;
                    height: 50px;
                    border-radius: 50%;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-size: 24px;
                    color: #667eea;
                '>
                    {st.session_state.username[0].upper() if st.session_state.username else '👤'}
                </div>
                <div style="color: white;">
                    <div style="font-size: 18px; font-weight: 600;">{st.session_state.username}</div>
                    <div style="font-size: 12px; opacity: 0.9;">
                        🕐 {st.session_state.login_time.strftime("%H:%M") if st.session_state.login_time else "now"}
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Navigation
    st.markdown("### Navigation")
    
    # Initialize session state for current page
    if "current_page" not in st.session_state:
        st.session_state.current_page = "movies"
    
    # Navigation buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🎬 Movies", use_container_width=True, type="primary" if st.session_state.current_page == "movies" else "secondary"):
            st.session_state.current_page = "movies"
            st.rerun()
    with col2:
        if st.button("👤 Profile", use_container_width=True, type="primary" if st.session_state.current_page == "profile" else "secondary"):
            st.session_state.current_page = "profile"
            st.rerun()
    
    st.markdown("---")
    
    # Only show filters if on movies page
    if st.session_state.current_page == "movies":
        st.header("🔍 Filters")
        # ... (seu código de filtros existente) ...
    else:
        # Logout button (shown on all pages)
        if st.button("🚪 Logout", use_container_width=True, type="primary"):
            for key in ['user_id', 'username', 'login_time', 'chat_messages', 'current_page']:
                if key in st.session_state:
                    st.session_state[key] = None if key != 'chat_messages' and key != 'current_page' else ([] if key == 'chat_messages' else 'movies')
            st.rerun()

# ==============================
# AI-POWERED CHATBOT
# ==============================

st.markdown("---")
st.subheader("🤖 AI Movie Assistant (Powered by Hugging Face)")

# Initialize chat session
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = [
        {"role": "assistant", "content": "👋 Hi! I'm your AI movie assistant. Ask me anything about movies!"}
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

    # Get AI response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = get_ai_response(user_input)
            st.write(response)
    
    st.session_state.chat_messages.append({"role": "assistant", "content": response})
