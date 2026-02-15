import pandas as pd

print("Lendo arquivos...")

basics = pd.read_csv("title.basics.tsv", sep="\t", low_memory=False)
ratings = pd.read_csv("title.ratings.tsv", sep="\t")

print("Filtrando apenas filmes...")

movies = basics[basics["titleType"] == "movie"]

print("Juntando com avaliações...")

movies = movies.merge(ratings, on="tconst")

print("Ordenando por nota...")

top100 = movies.sort_values(by="averageRating", ascending=False).head(100)

top100 = top100[[
    "primaryTitle",
    "startYear",
    "genres",
    "averageRating",
    "numVotes"
]]

top100.to_csv("top100_clean.csv", index=False)

print("Arquivo top100_clean.csv criado com sucesso!")
