import sqlite3
import pandas as pd

print("Lendo CSV...")
df = pd.read_csv("top100_clean.csv")

print("Conectando ao banco...")
conn = sqlite3.connect("movies.db")
cursor = conn.cursor()

print("Criando tabela...")
cursor.execute("""
CREATE TABLE IF NOT EXISTS movies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    year INTEGER,
    genre TEXT,
    rating REAL,
    votes INTEGER
)
""")

print("Inserindo dados...")

for _, row in df.iterrows():
    cursor.execute("""
    INSERT INTO movies (title, year, genre, rating, votes)
    VALUES (?, ?, ?, ?, ?)
    """, (
        row["primaryTitle"],
        int(row["startYear"]) if str(row["startYear"]).isdigit() else None,
        row["genres"],
        float(row["averageRating"]),
        int(row["numVotes"])
    ))

conn.commit()
conn.close()

print("Banco criado com sucesso!")
