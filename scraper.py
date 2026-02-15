import requests
from bs4 import BeautifulSoup
import pandas as pd

url = "https://www.imdb.com/chart/top"

headers = {
    "User-Agent": "Mozilla/5.0"
}

response = requests.get(url, headers=headers)

if response.status_code != 200:
    print("Erro ao acessar o site:", response.status_code)
    exit()

soup = BeautifulSoup(response.text, "html.parser")

movies = soup.select("tbody.lister-list tr")

data = []

for movie in movies[:100]:
    title = movie.select_one(".titleColumn a").text
    year = movie.select_one(".titleColumn span").text.strip("()")
    rating = movie.select_one(".ratingColumn.imdbRating strong").text
    
    data.append({
        "title": title,
        "year": int(year),
        "rating": float(rating)
    })

df = pd.DataFrame(data)
df.to_csv("top100_imdb.csv", index=False)

print("Arquivo top100_imdb.csv criado com sucesso!")
