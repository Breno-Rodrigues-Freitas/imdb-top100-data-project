import sqlite3

conn = sqlite3.connect("movies.db")
cursor = conn.cursor()

# Criar tabelas se não existirem
cursor.execute("""
CREATE TABLE IF NOT EXISTS genres (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS movie_genres (
    movie_id INTEGER,
    genre_id INTEGER,
    FOREIGN KEY(movie_id) REFERENCES movies(id),
    FOREIGN KEY(genre_id) REFERENCES genres(id)
)
""")

# Limpar tabelas caso já existam dados
cursor.execute("DELETE FROM movie_genres")
cursor.execute("DELETE FROM genres")

# Buscar filmes
cursor.execute("SELECT id, genre FROM movies")
movies = cursor.fetchall()

for movie_id, genre_string in movies:
    if genre_string:
        genres = [g.strip() for g in genre_string.split(",")]

        for genre in genres:
            cursor.execute("INSERT OR IGNORE INTO genres (name) VALUES (?)", (genre,))
            cursor.execute("SELECT id FROM genres WHERE name = ?", (genre,))
            genre_id = cursor.fetchone()[0]

            cursor.execute(
                "INSERT INTO movie_genres (movie_id, genre_id) VALUES (?, ?)",
                (movie_id, genre_id)
            )

conn.commit()
conn.close()

print("Database normalized successfully!")
