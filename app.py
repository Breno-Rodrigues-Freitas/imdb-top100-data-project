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
st.set_page_config(
    page_title="CineVault",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================
# GLOBAL CSS — CineVault Design
# ==============================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Variables ── */
:root {
    --bg:         #0a0a0f;
    --surface:    #111118;
    --surface2:   #1a1a24;
    --border:     #ffffff12;
    --accent:     #e8b84b;
    --accent2:    #c0392b;
    --text:       #e8e8f0;
    --muted:      #6b6b80;
    --radius:     12px;
    --font-head:  'Bebas Neue', sans-serif;
    --font-body:  'DM Sans', sans-serif;
}

/* ── Base ── */
html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--font-body) !important;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse 80% 40% at 50% -10%, #e8b84b18 0%, transparent 60%),
        var(--bg) !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }

/* ── Força sidebar sempre aberta e visível ── */
[data-testid="stSidebar"] {
    min-width: 260px !important;
    max-width: 320px !important;
    transform: none !important;
    visibility: visible !important;
}
[data-testid="stSidebarCollapsedControl"] { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }
section[data-testid="stSidebar"] > div { width: 100% !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] > div:first-child { padding-top: 1.5rem; }

/* ── Main title ── */
[data-testid="stAppViewContainer"] h1 {
    font-family: var(--font-head) !important;
    font-size: 3.2rem !important;
    letter-spacing: 3px !important;
    background: linear-gradient(90deg, var(--accent) 0%, #fff 60%);
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    margin-bottom: 0.25rem !important;
}

/* ── Subheaders ── */
h2, h3 {
    font-family: var(--font-head) !important;
    letter-spacing: 2px !important;
    color: var(--text) !important;
}

/* ── Tabs ── */
[data-testid="stTabs"] [role="tablist"] {
    background: var(--surface2) !important;
    border-radius: var(--radius) !important;
    padding: 4px !important;
    gap: 4px !important;
    border: 1px solid var(--border) !important;
}
[data-testid="stTabs"] button[role="tab"] {
    font-family: var(--font-body) !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    color: var(--muted) !important;
    border-radius: 8px !important;
    padding: 6px 16px !important;
    transition: all 0.2s !important;
    border: none !important;
}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    background: var(--accent) !important;
    color: #000 !important;
    font-weight: 600 !important;
}
[data-testid="stTabs"] [role="tabpanel"] {
    padding-top: 1.5rem !important;
}

/* ── Dataframe / Table ── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    overflow: hidden !important;
}
[data-testid="stDataFrame"] thead th {
    background: var(--surface2) !important;
    color: var(--accent) !important;
    font-family: var(--font-head) !important;
    letter-spacing: 1.5px !important;
    font-size: 0.8rem !important;
    border-bottom: 1px solid var(--border) !important;
}
[data-testid="stDataFrame"] tbody tr:nth-child(even) td {
    background: var(--surface2) !important;
}
[data-testid="stDataFrame"] tbody tr:hover td {
    background: #e8b84b12 !important;
}

/* ── Buttons ── */
[data-testid="stButton"] button {
    font-family: var(--font-body) !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    transition: all 0.2s !important;
    border: 1px solid var(--border) !important;
}
[data-testid="stButton"] button[kind="primary"] {
    background: var(--accent) !important;
    color: #000 !important;
    border-color: var(--accent) !important;
}
[data-testid="stButton"] button[kind="primary"]:hover {
    background: #f5c842 !important;
    box-shadow: 0 0 20px #e8b84b50 !important;
    transform: translateY(-1px) !important;
}
[data-testid="stButton"] button[kind="secondary"] {
    background: var(--surface2) !important;
    color: var(--text) !important;
}
[data-testid="stButton"] button[kind="secondary"]:hover {
    background: var(--surface) !important;
    border-color: var(--accent) !important;
    color: var(--accent) !important;
}

/* ── Inputs ── */
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: var(--font-body) !important;
    transition: border-color 0.2s !important;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stTextArea"] textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px #e8b84b20 !important;
}

/* ── Selectbox ── */
[data-testid="stSelectbox"] > div > div {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
}

/* ── Slider ── */
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
    background: var(--accent) !important;
    border-color: var(--accent) !important;
}
[data-testid="stSlider"] [data-baseweb="slider"] div[class*="Track"] {
    background: var(--surface2) !important;
}

/* ── Metrics ── */
[data-testid="stMetric"] {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 1rem 1.25rem !important;
    transition: border-color 0.2s !important;
}
[data-testid="stMetric"]:hover {
    border-color: var(--accent) !important;
}
[data-testid="stMetricLabel"] {
    color: var(--muted) !important;
    font-size: 0.75rem !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
}
[data-testid="stMetricValue"] {
    color: var(--accent) !important;
    font-family: var(--font-head) !important;
    font-size: 2rem !important;
    letter-spacing: 1px !important;
}

/* ── Alerts / Info / Success / Error ── */
[data-testid="stAlert"] {
    border-radius: var(--radius) !important;
    border: 1px solid var(--border) !important;
    background: var(--surface2) !important;
}

/* ── Forms ── */
[data-testid="stForm"] {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 1.5rem !important;
}

/* ── Spinner / Chat ── */
[data-testid="stChatMessage"] {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    margin-bottom: 0.75rem !important;
}
[data-testid="stChatInputContainer"] {
    background: var(--surface) !important;
    border-top: 1px solid var(--border) !important;
}
[data-testid="stChatInputContainer"] textarea {
    background: var(--surface2) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
}

/* ── Sidebar section labels ── */
[data-testid="stSidebar"] h3 {
    font-family: var(--font-head) !important;
    letter-spacing: 2px !important;
    font-size: 1.1rem !important;
    color: var(--muted) !important;
}

/* ── Dividers ── */
hr {
    border-color: var(--border) !important;
    margin: 1.5rem 0 !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--surface); }
::-webkit-scrollbar-thumb { background: var(--surface2); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent); }

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: var(--surface2) !important;
    border: 1px dashed var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 1rem !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--accent) !important;
}
</style>
""", unsafe_allow_html=True)

st.title("🎬 CineVault")

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
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT UNIQUE,
    password      TEXT,
    email         TEXT,
    profile_pic   TEXT,
    bio           TEXT,
    favorite_genre TEXT,
    joined_date   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")

# Add profile columns (if not exist) — log errors instead of silencing them
_profile_columns = {
    "email":          "TEXT DEFAULT ''",
    "profile_pic":    "TEXT DEFAULT ''",
    "bio":            "TEXT DEFAULT ''",
    "favorite_genre": "TEXT DEFAULT ''",
    "joined_date":    "TEXT DEFAULT ''",
}
for col, col_type in _profile_columns.items():
    try:
        conn.execute(f"ALTER TABLE users ADD COLUMN {col} {col_type}")
        conn.commit()
    except sqlite3.OperationalError:
        pass  # coluna já existe — comportamento esperado em toda execução após a primeira
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
            background: linear-gradient(135deg, #1a1a24 0%, #111118 100%);
            padding: 16px;
            border-radius: 12px;
            margin-bottom: 16px;
            border: 1px solid #ffffff12;
        ">
            <div style="display: flex; align-items: center; gap: 12px;">
                <img src="data:image/png;base64,{get_profile_pic_base64(profile_pic)}" 
                     style="width: 44px; height: 44px; border-radius: 50%; object-fit: cover; border: 2px solid #e8b84b;">
                <div>
                    <div style="font-family: 'Bebas Neue', sans-serif; font-size: 1.2rem; letter-spacing: 2px; color: #e8e8f0;">{st.session_state.username}</div>
                    <div style="font-size: 0.72rem; color: #6b6b80; letter-spacing: 0.5px;">
                        🕐 {st.session_state.login_time.strftime("%H:%M") if st.session_state.login_time else "now"}
                    </div>
                </div>
                <div style="margin-left: auto; width: 8px; height: 8px; border-radius: 50%; background: #2ecc71;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        initial = st.session_state.username[0].upper() if st.session_state.username else '?'
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #1a1a24 0%, #111118 100%);
            padding: 16px;
            border-radius: 12px;
            margin-bottom: 16px;
            border: 1px solid #ffffff12;
        ">
            <div style="display: flex; align-items: center; gap: 12px;">
                <div style="
                    background: linear-gradient(135deg, #e8b84b, #c0392b);
                    width: 44px; height: 44px;
                    border-radius: 50%;
                    display: flex; align-items: center; justify-content: center;
                    font-family: 'Bebas Neue', sans-serif;
                    font-size: 1.4rem; color: #000;
                    flex-shrink: 0;
                ">{initial}</div>
                <div>
                    <div style="font-family: 'Bebas Neue', sans-serif; font-size: 1.2rem; letter-spacing: 2px; color: #e8e8f0;">{st.session_state.username}</div>
                    <div style="font-size: 0.72rem; color: #6b6b80; letter-spacing: 0.5px;">
                        🕐 {st.session_state.login_time.strftime("%H:%M") if st.session_state.login_time else "now"}
                    </div>
                </div>
                <div style="margin-left: auto; width: 8px; height: 8px; border-radius: 50%; background: #2ecc71;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Navigation
    st.markdown("### Navigation")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🎬 Movies", use_container_width=True, type="primary" if st.session_state.current_page == "movies" else "secondary"):
            st.session_state.current_page = "movies"
            st.rerun()
    with col2:
        if st.button("🕐 History", use_container_width=True, type="primary" if st.session_state.current_page == "history" else "secondary"):
            st.session_state.current_page = "history"
            st.rerun()
    with col3:
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

    # Migração defensiva: garante cada coluna antes de qualquer query
    _migration_columns = {
        "email":          "TEXT DEFAULT ''",
        "profile_pic":    "TEXT DEFAULT ''",
        "bio":            "TEXT DEFAULT ''",
        "favorite_genre": "TEXT DEFAULT ''",
        "joined_date":    "TEXT DEFAULT ''",
    }
    for col, col_type in _migration_columns.items():
        try:
            conn.execute(f"ALTER TABLE users ADD COLUMN {col} {col_type}")
            conn.commit()
        except sqlite3.OperationalError:
            pass  # coluna já existe — comportamento esperado

    try:
        user_data = conn.execute("""
            SELECT
                username,
                COALESCE(email, '')          AS email,
                COALESCE(profile_pic, '')    AS profile_pic,
                COALESCE(bio, '')            AS bio,
                COALESCE(favorite_genre, '') AS favorite_genre,
                COALESCE(joined_date, '')    AS joined_date
            FROM users
            WHERE id = ?
        """, (st.session_state.user_id,)).fetchone()
    except Exception as e:
        logger.error("Error fetching user data: %s", e)
        st.error(f"Could not load profile data. Details: {e}")
        return
    
    if user_data:
        username, email, profile_pic, bio, favorite_genre, joined_date = user_data
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### Profile Picture")
            if profile_pic and os.path.exists(profile_pic):
                st.image(profile_pic, width=200, caption="Current Photo")
            else:
                initial = username[0].upper() if username else '?'
                st.markdown(f"""
                <div style='
                    width: 160px; height: 160px;
                    border-radius: 50%;
                    background: linear-gradient(135deg, #e8b84b, #c0392b);
                    display: flex; align-items: center; justify-content: center;
                    font-family: "Bebas Neue", sans-serif;
                    font-size: 72px; color: #000;
                    margin-bottom: 20px;
                    border: 3px solid #ffffff12;
                '>
                    {initial}
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
# HISTORY PAGE
# ==============================
def history_page():
    st.subheader("🕐 Search History")

    # ── Controles ──────────────────────────────────────────────
    col_search, col_sort, col_clear = st.columns([3, 2, 1])
    with col_search:
        filter_text = st.text_input("Filter by title", placeholder="Type to filter...", label_visibility="collapsed")
    with col_sort:
        sort_order = st.selectbox("Order", ["Newest first", "Oldest first", "A → Z", "Z → A"], label_visibility="collapsed")
    with col_clear:
        if st.button("🗑️ Clear all", use_container_width=True, type="secondary"):
            try:
                conn.execute("DELETE FROM history WHERE user_id = ?", (st.session_state.user_id,))
                conn.commit()
                st.success("History cleared.")
                st.rerun()
            except Exception as e:
                logger.error("Error clearing history: %s", e)
                st.error("Could not clear history.")

    st.markdown("---")

    # ── Buscar registros ───────────────────────────────────────
    try:
        order_sql = {
            "Newest first": "searched_at DESC",
            "Oldest first": "searched_at ASC",
            "A → Z":        "movie_title ASC",
            "Z → A":        "movie_title DESC",
        }[sort_order]

        history_df = pd.read_sql(f"""
            SELECT
                id,
                movie_title,
                searched_at
            FROM history
            WHERE user_id = ?
            ORDER BY {order_sql}
        """, conn, params=(st.session_state.user_id,))
    except Exception as e:
        logger.error("Error fetching history: %s", e)
        st.error("Could not load history.")
        return

    # ── Filtro por texto ───────────────────────────────────────
    if filter_text:
        history_df = history_df[
            history_df["movie_title"].str.contains(filter_text, case=False, na=False)
        ]

    # ── Resumo ─────────────────────────────────────────────────
    total = len(history_df)
    unique = history_df["movie_title"].nunique() if not history_df.empty else 0

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("📋 Total Searches", total)
    with m2:
        st.metric("🎬 Unique Movies", unique)
    with m3:
        limit_pct = f"{total}/{HISTORY_LIMIT}"
        st.metric("📦 History Usage", limit_pct)

    st.markdown("---")

    # ── Lista de itens ─────────────────────────────────────────
    if history_df.empty:
        st.markdown("""
        <div style="
            text-align: center;
            padding: 3rem;
            color: #6b6b80;
            border: 1px dashed #ffffff12;
            border-radius: 12px;
            margin-top: 1rem;
        ">
            <div style="font-size: 3rem; margin-bottom: 0.5rem;">🎬</div>
            <div style="font-family: 'Bebas Neue', sans-serif; font-size: 1.4rem; letter-spacing: 2px;">
                No searches yet
            </div>
            <div style="font-size: 0.85rem; margin-top: 0.25rem;">
                Start searching for movies to build your history
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    # Renderizar cada item como card com botão de deletar e re-buscar
    for _, row in history_df.iterrows():
        record_id   = row["id"]
        movie_title = row["movie_title"]
        searched_at = row["searched_at"]

        # Formatar data
        try:
            dt = datetime.strptime(searched_at[:19], "%Y-%m-%d %H:%M:%S")
            date_str = dt.strftime("%d %b %Y · %H:%M")
        except Exception:
            date_str = searched_at

        col_info, col_search_btn, col_del = st.columns([6, 1, 1])

        with col_info:
            st.markdown(f"""
            <div style="
                background: #1a1a24;
                border: 1px solid #ffffff12;
                border-radius: 10px;
                padding: 12px 16px;
                display: flex;
                align-items: center;
                gap: 12px;
            ">
                <div style="
                    background: linear-gradient(135deg, #e8b84b22, #c0392b22);
                    border: 1px solid #e8b84b33;
                    border-radius: 8px;
                    width: 36px; height: 36px;
                    display: flex; align-items: center; justify-content: center;
                    font-size: 1.1rem; flex-shrink: 0;
                ">🎬</div>
                <div>
                    <div style="font-weight: 600; color: #e8e8f0; font-size: 0.95rem;">{movie_title}</div>
                    <div style="font-size: 0.75rem; color: #6b6b80; margin-top: 2px;">🕐 {date_str}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_search_btn:
            if st.button("🔍", key=f"search_{record_id}", help=f"Search '{movie_title}' again", use_container_width=True):
                st.session_state.current_page = "movies"
                st.session_state["search_title"] = movie_title
                st.rerun()

        with col_del:
            if st.button("✕", key=f"del_{record_id}", help="Remove this entry", use_container_width=True):
                try:
                    conn.execute("DELETE FROM history WHERE id = ? AND user_id = ?", (record_id, st.session_state.user_id))
                    conn.commit()
                    st.rerun()
                except Exception as e:
                    logger.error("Error deleting history entry %s: %s", record_id, e)

    # ── Paginação simples via expander se lista grande ─────────
    if total > 20:
        st.caption(f"Showing {min(total, 20)} of {total} entries. Use the filter above to narrow results.")


# ==============================
# PAGE ROUTING
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
    if st.session_state.current_page == "history":
        history_page()
    else:
        profile_page()