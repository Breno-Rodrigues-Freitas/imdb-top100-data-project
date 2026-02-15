import pandas as pd

# Carregar dados
df = pd.read_csv("top100_clean.csv")

print("Gêneros disponíveis no Top 100:\n")

# Explodir múltiplos gêneros
df["genres"] = df["genres"].str.split(",")
df_exploded = df.explode("genres")

genres_list = sorted(df_exploded["genres"].unique())

for g in genres_list:
    print("-", g)

print("\nDigite um gênero exatamente como aparece acima:")
user_genre = input("> ")

filtered = df_exploded[df_exploded["genres"] == user_genre]

if filtered.empty:
    print("Nenhum filme encontrado para esse gênero.")
else:
    print("\n🎬 Recomendações:")
    recommendations = filtered.sort_values(by="averageRating", ascending=False).head(5)
    print(recommendations[["primaryTitle", "startYear", "averageRating"]])
