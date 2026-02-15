import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# =====================
# Carregar dados
# =====================
df = pd.read_csv("top100_clean.csv")

df["genres"] = df["genres"].str.split(",")
df["startYear"] = pd.to_numeric(df["startYear"], errors="coerce")
df["averageRating"] = pd.to_numeric(df["averageRating"], errors="coerce")

# Remover valores nulos
df = df.dropna(subset=["genres", "startYear", "averageRating"])

# =====================
# Transformar gêneros em vetores
# =====================
mlb = MultiLabelBinarizer()
genres_encoded = mlb.fit_transform(df["genres"])

genres_df = pd.DataFrame(genres_encoded, columns=mlb.classes_)

# =====================
# Criar matriz final de features
# =====================
features = pd.concat([
    genres_df,
    df[["averageRating", "startYear"]].reset_index(drop=True)
], axis=1)

# Normalizar dados numéricos
scaler = StandardScaler()
features[["averageRating", "startYear"]] = scaler.fit_transform(
    features[["averageRating", "startYear"]]
)

# =====================
# Treinar modelo KNN
# =====================
model = NearestNeighbors(n_neighbors=6, metric="euclidean")
model.fit(features)

# =====================
# Input usuário
# =====================
print("Digite parte do nome do filme:")
movie_name = input("> ")

matches = df[df["primaryTitle"].str.contains(movie_name, case=False, na=False)]

if matches.empty:
    print("Filme não encontrado.")
    exit()

movie_index = matches.index[0]

# Encontrar vizinhos mais próximos
distances, indices = model.kneighbors([features.iloc[movie_index]])

print("\n🎬 Recomendações ML:\n")

for idx in indices[0][1:]:
    print(f"- {df.iloc[idx]['primaryTitle']} ({int(df.iloc[idx]['startYear'])})")
