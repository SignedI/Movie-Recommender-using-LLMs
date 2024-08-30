pip install requests pandas

import pandas as pd
import requests
from google.colab import files

API_KEY = ''

netflix_df = pd.read_csv('netflix_titles.csv')

def get_imdb_data(movie_id, title):
    url = f"http://www.omdbapi.com/?t={title}&apikey={API_KEY}"
    response = requests.get(url)
    data = response.json()
    if data['Response'] == 'True':
        return {
            'imdbRating': data.get('imdbRating', 'N/A'),
            'imdbVotes': data.get('imdbVotes', 'N/A')
        }
    else:
        return {
            'imdbRating': 'N/A',
            'imdbVotes': 'N/A'
        }


netflix_df['imdbRating'] = ''
netflix_df['imdbVotes'] = ''

for index, row in netflix_df.iterrows():
    title = row['title']
    imdb_data = get_imdb_data(row['id'], title)  # Adjust the column names as needed
    netflix_df.at[index, 'imdbRating'] = imdb_data['imdbRating']
    netflix_df.at[index, 'imdbVotes'] = imdb_data['imdbVotes']

output_file = 'netflix_with_ratings.csv'
netflix_df.to_csv(output_file, index=False)


