import streamlit as st
import pandas as pd
import sqlite3
import requests
import os
import hashlib
import hmac
import re
import base64
import logging
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==============================
# LOGGING CONFIG
# ==============================
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

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
    conn = sqlite3.connect("movies.db", check_same_thread=False)
    # Enable WAL mode for better concurrency
    conn.execute("PRAGMA journal_mode=WAL")
    return conn

conn = get_connection()

# ==============================
# CREATE TABLES (with profile columns)
# ==============================
conn.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE,
    password TEXT
)
""")

# Add profile columns (if not exist) — log errors instead of silencing them
_profile_columns = {
    "email": "TEXT",
    "profile_pic": "TEXT",
    "bio": "TEXT",
    "favorite_genre": "TEXT",
    "joined_date": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
}
for col, col_type in _profile_columns.items():
    try:
        conn.execute(f"ALTER TABLE users ADD COLUMN {col} {col_type}")
    except sqlite3.OperationalError:
        pass  # Column already exists — expected on every run after the first
    except Exception as e:
        logger.error("Error adding column '%s': %s", col, e)

conn.execute("""
CREATE TABLE IF NOT EXISTS history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    movie_title TEXT,
    searched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(user_id) REFERENCES users(id)
)
""")

conn.execute("""
CREATE TABLE IF NOT EXISTS login_attempts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT,
    ip_address TEXT,
    attempt_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    success BOOLEAN
)
""")

# Indexes for performance
conn.execute("CREATE INDEX IF NOT EXISTS idx_login_attempts_username ON login_attempts(username)")
conn.execute("CREATE INDEX IF NOT EXISTS idx_login_attempts_time ON login_attempts(attempt_time)")
conn.commit()

# ==============================
# SECURITY CONSTANTS
# ==============================
MIN_PASSWORD_LENGTH = 6
MAX_LOGIN_ATTEMPTS = 5
LOCKOUT_TIME = 15  # minutes

# ==============================
# API / HISTORY CONSTANTS
# ==============================
API_MAX_RETRIES = 3        # tentativas máximas para chamadas de API
API_RETRY_DELAY = 1.5      # segundos entre tentativas (backoff fixo)
HISTORY_LIMIT = 100        # máximo de registros de histórico por usuário

# ==============================
# PASSWORD HASHING FUNCTIONS
# ==============================
def hash_password(password: str) -> str:
    salt = os.urandom(32).hex()
    hash_obj = hashlib.pbkdf2_hmac(
        'sha256',
        password.encode('utf-8'),
        salt.encode('utf-8'),
        100000
    )
    return f"{salt}${hash_obj.hex()}"

def verify_password(password: str, hashed: str) -> bool:
    try:
        salt, hash_value = hashed.split('$')
        hash_obj = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000
        )
        return hmac.compare_digest(hash_obj.hex(), hash_value)
    except ValueError as e:
        logger.error("Error verifying password (invalid hash format): %s", e)
        return False
    except Exception as e:
        logger.error("Unexpected error verifying password: %s", e)
        return False

def validate_username(username: str) -> tuple[bool, str]:
    if not username or len(username) < 3:
        return False, "Username must have at least 3 characters"
    if len(username) > 30:
        return False, "Username must have at most 30 characters"
    if not re.match("^[a-zA-Z0-9_]+$", username):
        return False, "Username can only contain letters, numbers and underscore"
    return True, "Valid username"

def validate_password(password: str) -> tuple[bool, str]:
    if len(password) < MIN_PASSWORD_LENGTH:
        return False, f"Password must have at least {MIN_PASSWORD_LENGTH} characters"
    if len(password) > 50:
        return False, "Password must have at most 50 characters"
    return True, "Valid password"

# ==============================
# LOGIN ATTEMPTS TRACKING
# ==============================
def check_login_attempts(username: str) -> tuple[bool, str]:
    try:
        cutoff = (datetime.now() - timedelta(minutes=LOCKOUT_TIME)).strftime("%Y-%m-%d %H:%M:%S")
        attempts = conn.execute("""
            SELECT COUNT(*) FROM login_attempts 
            WHERE username = ? AND success = 0 AND attempt_time > ?
        """, (username, cutoff)).fetchone()[0]
        if attempts >= MAX_LOGIN_ATTEMPTS:
            return False, f"Too many failed attempts. Please try again in {LOCKOUT_TIME} minutes."
        return True, ""
    except Exception as e:
        logger.error("Error checking login attempts for '%s': %s", username, e)
        return True, ""

def record_login_attempt(username: str, success: bool):
    try:
        conn.execute(
            "INSERT INTO login_attempts (username, ip_address, success) VALUES (?, ?, ?)",
            (username, "unknown", success)
        )
        conn.commit()
    except Exception as e:
        logger.error("Error recording login attempt for '%s': %s", username, e)

# ==============================
# PROFILE FUNCTIONS
# ==============================
def save_profile_picture(uploaded_file, username):
    if uploaded_file is not None:
        try:
            os.makedirs("profile_pics", exist_ok=True)
            file_extension = uploaded_file.name.split('.')[-1]
            filename = f"profile_pics/{username}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{file_extension}"
            with open(filename, "wb") as f:
                f.write(uploaded_file.getbuffer())
            return filename
        except Exception as e:
            logger.error("Error saving profile picture for '%s': %s", username, e)
    return None

def get_profile_pic_base64(image_path):
    try:
        if image_path and os.path.exists(image_path):
            with open(image_path, "rb") as f:
                return base64.b64encode(f.read()).decode()
    except Exception as e:
        logger.error("Error reading profile picture at '%s': %s", image_path, e)
    return None

# ==============================
# SESSION STATE INIT
# ==============================
if "user_id" not in st.session_state:
    st.session_state.user_id = None
    st.session_state.username = None
    st.session_state.login_time = None
    st.session_state.remember_me = False
    st.session_state.current_page = "movies"
    st.session_state.chat_messages = [
        {"role": "assistant", "content": "👋 Hi! I'm your AI movie assistant. Ask me anything about movies!"}
    ]

# ==============================
# REMEMBER ME — restore session from cookie-like query param
# ==============================
# NOTE: Streamlit doesn't support real persistent cookies. We simulate
# "remember me" by extending the session duration check. If login_time
# is set and the user chose remember_me, we allow up to 7 days before
# forcing re-login; otherwise the session is valid for 1 day.
def is_session_expired() -> bool:
    if st.session_state.login_time is None:
        return True
    max_duration = timedelta(days=7) if st.session_state.remember_me else timedelta(days=1)
    return datetime.now() - st.session_state.login_time > max_duration

if st.session_state.user_id is not None and is_session_expired():
    st.warning("Your session has expired. Please log in again.")
    for key in ['user_id', 'username', 'login_time', 'remember_me', 'current_page', 'chat_messages']:
        st.session_state[key] = None if key not in ('current_page', 'chat_messages', 'remember_me') else (
            'movies' if key == 'current_page' else ([] if key == 'chat_messages' else False)
        )
    st.rerun()

# ==============================
# AUTH UI (if not logged in)
# ==============================
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
        
        # ----- LOGIN -----
        with tab1:
            with st.form("login_form"):
                username = st.text_input("Username", placeholder="Enter your username")
                password = st.text_input("Password", type="password", placeholder="Enter your password")
                col1, col2 = st.columns([1,1])
                with col1:
                    remember_me = st.checkbox("Remember me (7 days)")
                with col2:
                    st.write("")
                submitted = st.form_submit_button("🚀 Login", use_container_width=True)
                
                if submitted:
                    if not username or not password:
                        st.error("Please fill in all fields")
                    else:
                        can_login, msg = check_login_attempts(username)
                        if not can_login:
                            st.error(msg)
                        else:
                            user = conn.execute(
                                "SELECT id, username, password FROM users WHERE username = ?",
                                (username,)
                            ).fetchone()
                            if user and verify_password(password, user[2]):
                                record_login_attempt(username, True)
                                st.session_state.user_id = user[0]
                                st.session_state.username = user[1]
                                st.session_state.login_time = datetime.now()
                                st.session_state.remember_me = remember_me
                                st.success(f"Welcome back, {username}! 🎉")
                                st.rerun()
                            else:
                                record_login_attempt(username, False)
                                st.error("Invalid username or password")
        
        # ----- REGISTER -----
        with tab2:
            with st.form("register_form"):
                new_username = st.text_input("Choose username", placeholder="Min 3 characters, letters/numbers/_")
                new_password = st.text_input("Choose password", type="password", placeholder=f"Min {MIN_PASSWORD_LENGTH} characters")
                confirm_password = st.text_input("Confirm password", type="password")
                
                if new_password:
                    strength = "Weak"
                    color = "red"
                    if len(new_password) >= 8 and any(c.isdigit() for c in new_password) and any(c.isupper() for c in new_password):
                        strength = "Strong"
                        color = "green"
                    elif len(new_password) >= 6:
                        strength = "Medium"
                        color = "orange"
                    st.markdown(f"Password strength: <span style='color:{color}'>{strength}</span>", unsafe_allow_html=True)
                
                submitted = st.form_submit_button("✨ Create Account", use_container_width=True)
                
                if submitted:
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
                            existing = conn.execute("SELECT id FROM users WHERE username = ?", (new_username,)).fetchone()
                            if existing:
                                st.error("Username already exists")
                            else:
                                hashed = hash_password(new_password)
                                conn.execute(
                                    "INSERT INTO users (username, password, joined_date) VALUES (?, ?, ?)",
                                    (new_username, hashed, datetime.now())
                                )
                                conn.commit()
                                st.success("✅ Account created successfully! Please login.")
                        except Exception as e:
                            logger.error("Error creating account for '%s': %s", new_username, e)
                            st.error(f"Error creating account: {str(e)}")
        
        # ----- FORGOT PASSWORD -----
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
                        # Always show same message to avoid username enumeration
                        st.success("📧 If the username exists, instructions will be sent.")
                        st.info("(Email functionality not implemented in this demo)")
        
        st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# ==============================
# SIDEBAR (USER INFO & NAVIGATION)
# ==============================
with st.sidebar:
    try:
        user_data = conn.execute(
            "SELECT profile_pic FROM users WHERE id = ?",
            (st.session_state.user_id,)
        ).fetchone()
        profile_pic = user_data[0] if user_data else None
    except Exception as e:
        logger.error("Error fetching profile pic: %s", e)
        profile_pic = None

    # Profile card
    if profile_pic and os.path.exists(profile_pic):
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            border-radius: 15px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        ">
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
                    color: #667eea;
                ">
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
    
    # Filters (only on movies page)
    if st.session_state.current_page == "movies":
        st.header("🔍 Filters")

        # FIX: query correta — gêneros ficam na tabela `genres`, não em `movies`
        try:
            genres = pd.read_sql(
                "SELECT DISTINCT name FROM genres ORDER BY name", conn
            )["name"].tolist()
        except Exception as e:
            logger.error("Error fetching genres: %s", e)
            genres = []
        
        search_title = st.text_input("Search movie title", key="search_title")
        selected_genre = st.selectbox("Select genre", ["All"] + genres, key="selected_genre")
        min_rating = st.slider("Minimum rating", 0.0, 10.0, 8.0, key="min_rating")
    else:
        search_title = ""
        selected_genre = "All"
        min_rating = 0.0
        if st.button("🚪 Logout", use_container_width=True, type="primary"):
            for key in ['user_id', 'username', 'login_time', 'remember_me', 'chat_messages', 'current_page']:
                if key == 'chat_messages':
                    st.session_state[key] = []
                elif key == 'current_page':
                    st.session_state[key] = 'movies'
                elif key == 'remember_me':
                    st.session_state[key] = False
                else:
                    st.session_state[key] = None
            st.rerun()

# ==============================
# HISTORY HELPERS
# ==============================
def save_to_history(user_id: int, movie_title: str):
    """Salva busca no histórico e garante que o limite por usuário seja respeitado."""
    try:
        conn.execute(
            "INSERT INTO history (user_id, movie_title) VALUES (?, ?)",
            (user_id, movie_title)
        )
        # Remove os registros mais antigos que ultrapassem o limite
        conn.execute("""
            DELETE FROM history
            WHERE user_id = ?
              AND id NOT IN (
                  SELECT id FROM history
                  WHERE user_id = ?
                  ORDER BY searched_at DESC
                  LIMIT ?
              )
        """, (user_id, user_id, HISTORY_LIMIT))
        conn.commit()
    except Exception as e:
        logger.error("Error saving search history for user %s: %s", user_id, e)


# ==============================
# HTTP RETRY HELPER
# ==============================
def _request_with_retry(method: str, url: str, **kwargs) -> requests.Response | None:
    """Faz uma requisição HTTP com retry exponencial.
    
    Tenta até API_MAX_RETRIES vezes em caso de erro de rede ou status 5xx.
    Retorna None se todas as tentativas falharem.
    """
    for attempt in range(1, API_MAX_RETRIES + 1):
        try:
            response = requests.request(method, url, **kwargs)
            if response.status_code < 500:
                # Qualquer resposta que não seja erro de servidor é retornada
                # (erros 4xx como 404 são válidos e não devem ser retentados)
                return response
            logger.warning(
                "Attempt %d/%d — server error %s for %s",
                attempt, API_MAX_RETRIES, response.status_code, url
            )
        except requests.ConnectionError as e:
            logger.warning("Attempt %d/%d — connection error: %s", attempt, API_MAX_RETRIES, e)
        except requests.Timeout as e:
            logger.warning("Attempt %d/%d — timeout: %s", attempt, API_MAX_RETRIES, e)

        if attempt < API_MAX_RETRIES:
            time.sleep(API_RETRY_DELAY * attempt)  # backoff: 1.5s, 3s

    logger.error("All %d attempts failed for %s", API_MAX_RETRIES, url)
    return None


# ==============================
# OMDB FETCH FUNCTION
# ==============================
def fetch_movie_from_api(title):
    if not OMDB_API_KEY:
        st.error("OMDB_API_KEY not found.")
        return None

    # requests.get com params faz URL encoding automático do título
    response = _request_with_retry(
        "GET",
        "http://www.omdbapi.com/",
        params={"t": title, "type": "movie", "apikey": OMDB_API_KEY},
        timeout=10,
    )

    if response is None:
        st.error("Could not reach OMDB API after several attempts. Please try again later.")
        return None

    try:
        data = response.json()
    except ValueError:
        logger.error("OMDB returned non-JSON response: %s", response.text[:200])
        st.error("Unexpected response from OMDB API.")
        return None

    if data.get("Response") == "True":
        runtime = data.get("Runtime", "")
        if runtime not in ("N/A", "") and "min" in runtime:
            try:
                if int(runtime.replace(" min", "")) < 30:
                    st.warning("This appears to be a short film (<30min).")
            except ValueError:
                pass
        return {
            "Title": data.get("Title"),
            "Year": data.get("Year"),
            "Genre": data.get("Genre"),
            "Rating": data.get("imdbRating"),
            "Votes": data.get("imdbVotes"),
            "Plot": data.get("Plot"),
        }

    logger.info("OMDB found no result for title: '%s'", title)
    return None


# ==============================
# YOUTUBE TRAILER FUNCTION
# ==============================
def fetch_trailer(title):
    if not YOUTUBE_API_KEY:
        return None

    response = _request_with_retry(
        "GET",
        "https://www.googleapis.com/youtube/v3/search",
        params={
            "part": "snippet",
            "q": f"{title} official trailer",
            "key": YOUTUBE_API_KEY,
            "type": "video",
            "maxResults": 1,
        },
        timeout=10,
    )

    if response is None:
        logger.error("YouTube API unreachable after retries for title: '%s'", title)
        return None

    try:
        data = response.json()
        items = data.get("items", [])
        if items:
            return items[0]["id"]["videoId"]
    except (ValueError, KeyError) as e:
        logger.error("Error parsing YouTube response: %s", e)

    return None

# ==============================
# HUGGING FACE AI FUNCTIONS
# ==============================
def get_movie_context():
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
    except Exception as e:
        logger.error("Error fetching movie context: %s", e)
        return "Database not available."

def get_ai_response(user_message):
    if not HUGGINGFACE_API_KEY:
        return "AI service not configured. Please check your HUGGINGFACE_API_KEY."
    try:
        API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"
        headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
        movie_context = get_movie_context()
        prompt = f"""<s>[INST] You are a movie assistant. Here are some movies in the database:
{movie_context}

Answer questions about movies. Keep answers short and helpful.

User: {user_message} [/INST]"""
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
            if isinstance(result, list) and len(result) > 0:
                if isinstance(result[0], dict) and 'generated_text' in result[0]:
                    return result[0]['generated_text'].split('[/INST]')[-1].strip()
                return str(result[0])
            elif isinstance(result, dict) and 'generated_text' in result:
                return result['generated_text'].split('[/INST]')[-1].strip()
            return "I received a response but couldn't understand it."
        elif response.status_code == 503:
            return "The AI model is loading. Please try again in 10 seconds."
        else:
            logger.error("HuggingFace API error %s: %s", response.status_code, response.text)
            return f"Error {response.status_code}. Please try again."
    except requests.exceptions.Timeout:
        return "Request timed out. Please try again."
    except Exception as e:
        logger.error("Unexpected error in get_ai_response: %s", e)
        return f"Error: {str(e)}"

# ==============================
# PROFILE PAGE
# ==============================
def profile_page():
    st.subheader("👤 Your Profile")
    
    try:
        user_data = conn.execute(
            "SELECT username, email, profile_pic, bio, favorite_genre, joined_date FROM users WHERE id = ?",
            (st.session_state.user_id,)
        ).fetchone()
    except Exception as e:
        logger.error("Error fetching user data: %s", e)
        st.error("Could not load profile data.")
        return
    
    if user_data:
        username, email, profile_pic, bio, favorite_genre, joined_date = user_data
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### Profile Picture")
            if profile_pic and os.path.exists(profile_pic):
                st.image(profile_pic, width=200, caption="Current Photo")
            else:
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
            
            uploaded_file = st.file_uploader("Change profile picture", type=['png', 'jpg', 'jpeg', 'gif'], key="profile_pic_uploader")
            if uploaded_file is not None:
                st.image(uploaded_file, width=150, caption="Preview")
                if st.button("💾 Save New Picture", use_container_width=True):
                    file_path = save_profile_picture(uploaded_file, username)
                    if file_path:
                        try:
                            conn.execute("UPDATE users SET profile_pic = ? WHERE id = ?", (file_path, st.session_state.user_id))
                            conn.commit()
                            st.success("Profile picture updated!")
                            st.rerun()
                        except Exception as e:
                            logger.error("Error saving profile pic to DB: %s", e)
                            st.error("Failed to save picture.")
        
        with col2:
            st.markdown("### Profile Information")
            with st.form("profile_form"):
                st.text_input("Username", value=username, disabled=True)
                new_email = st.text_input("Email", value=email if email else "", placeholder="your@email.com")
                new_bio = st.text_area("Bio", value=bio if bio else "", placeholder="Tell us about your movie preferences...", height=100)
                
                try:
                    genres_df = pd.read_sql("SELECT name FROM genres ORDER BY name", conn)
                    genre_list = ["None"] + genres_df["name"].tolist() if not genres_df.empty else ["None"]
                except Exception as e:
                    logger.error("Error fetching genres for profile: %s", e)
                    genre_list = ["None"]

                current_genre = favorite_genre if favorite_genre in genre_list else "None"
                new_genre = st.selectbox("Favorite Genre", genre_list, index=genre_list.index(current_genre) if current_genre in genre_list else 0)
                
                if joined_date:
                    st.text_input("Member Since", value=joined_date[:10], disabled=True)
                
                submitted = st.form_submit_button("💾 Update Profile", use_container_width=True)
                if submitted:
                    try:
                        conn.execute(
                            "UPDATE users SET email = ?, bio = ?, favorite_genre = ? WHERE id = ?",
                            (new_email, new_bio, new_genre if new_genre != "None" else None, st.session_state.user_id)
                        )
                        conn.commit()
                        st.success("Profile updated successfully!")
                        st.rerun()
                    except Exception as e:
                        logger.error("Error updating profile: %s", e)
                        st.error("Failed to update profile.")
    
    # Stats
    st.markdown("---")
    st.subheader("📊 Your Movie Stats")
    col1, col2, col3 = st.columns(3)
    
    try:
        searches = conn.execute("SELECT COUNT(*) FROM history WHERE user_id = ?", (st.session_state.user_id,)).fetchone()[0]
    except Exception as e:
        logger.error("Error fetching search count: %s", e)
        searches = 0

    with col1:
        st.metric("🔍 Total Searches", searches)
    
    try:
        # Query otimizada: busca direto no histórico + movie_genres sem o LIKE custoso.
        # Usa cache de 5 minutos para não re-executar a cada re-render do perfil.
        @st.cache_data(ttl=300)
        def get_favorite_genre(user_id: int) -> str:
            row = conn.execute("""
                SELECT g.name
                FROM history h
                JOIN movies m ON m.title = h.movie_title
                JOIN movie_genres mg ON mg.movie_id = m.id
                JOIN genres g ON g.id = mg.genre_id
                WHERE h.user_id = ?
                GROUP BY g.name
                ORDER BY COUNT(*) DESC
                LIMIT 1
            """, (user_id,)).fetchone()
            return row[0] if row else "N/A"

        fav_genre_name = get_favorite_genre(st.session_state.user_id)
        with col2:
            st.metric("🎯 Favorite Genre", fav_genre_name)
    except Exception as e:
        logger.error("Error fetching favorite genre: %s", e)
        with col2:
            st.metric("🎯 Favorite Genre", "N/A")
    
    try:
        movies_count = conn.execute("SELECT COUNT(*) FROM movies").fetchone()[0]
    except Exception as e:
        logger.error("Error fetching movies count: %s", e)
        movies_count = 0

    with col3:
        st.metric("📚 Total Movies", movies_count)

# ==============================
# MAIN CONTENT
# ==============================
if st.session_state.current_page == "movies":
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

    try:
        df = pd.read_sql(query, conn, params=params)
    except Exception as e:
        logger.error("Error running movies query: %s", e)
        df = pd.DataFrame()
    
    tab1, tab2, tab3, tab4 = st.tabs(["🎥 Movies", "🏆 Top Rated", "🎯 Similar Movies", "🤖 Chat Assistant"])
    
    with tab1:
        st.subheader("🎥 Movie Results")
        if df.empty and search_title:
            st.info("Movie not found in local database. Searching online... 🌐")
            api_movie = fetch_movie_from_api(search_title)
            if api_movie:
                # Salva no histórico com controle de limite
                save_to_history(st.session_state.user_id, search_title)
                
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
                    try:
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
                            st.info("Movie already exists in the database.")
                    except Exception as e:
                        logger.error("Error adding movie to DB: %s", e)
                        st.error("Failed to add movie to database.")
            else:
                st.error("Movie not found online either.")
        elif not df.empty:
            # Salva no histórico apenas quando há busca nova (não em toda re-renderização)
            if search_title and st.session_state.get("last_search") != search_title:
                save_to_history(st.session_state.user_id, search_title)
                st.session_state.last_search = search_title
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No movies found with current filters")
    
    with tab2:
        st.subheader("🏆 Top Rated Movies")
        try:
            top_df = pd.read_sql("""
                SELECT title, year, rating, votes
                FROM movies
                WHERE rating IS NOT NULL
                ORDER BY rating DESC, votes DESC
                LIMIT 10
            """, conn)
            st.dataframe(top_df, use_container_width=True)
        except Exception as e:
            logger.error("Error fetching top rated movies: %s", e)
            st.error("Could not load top rated movies.")
    
    with tab3:
        st.subheader("🎯 Find Similar Movies")
        try:
            sim_query = """
            SELECT 
                m.id,
                m.title,
                GROUP_CONCAT(g.name, ' ') as genres
            FROM movies m
            JOIN movie_genres mg ON m.id = mg.movie_id
            JOIN genres g ON g.id = mg.genre_id
            GROUP BY m.id
            """
            movies_df = pd.read_sql(sim_query, conn)
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
            else:
                st.info("Add movies to the database to see recommendations")
        except Exception as e:
            logger.error("Error in similar movies: %s", e)
            st.error("Could not load similarity data.")
    
    with tab4:
        st.subheader("🤖 AI Movie Assistant (Powered by Hugging Face)")
        for message in st.session_state.chat_messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        if user_input := st.chat_input("Ask about movies..."):
            st.session_state.chat_messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.write(user_input)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = get_ai_response(user_input)
                    st.write(response)
            st.session_state.chat_messages.append({"role": "assistant", "content": response})

else:
    profile_page()