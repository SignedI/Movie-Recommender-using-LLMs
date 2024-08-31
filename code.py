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

data = pd.read_csv("netflix_with_ratings.csv")

# Concatenate the features into a single text field for embedding
data['combined'] = data['movie_title'] + " - " + data['cast'] + " - " + \
                   data['director'] + " - " + data['release_date'] + " - Rating: " + data['rating'].astype(str)


from openai.embeddings_utils import get_embedding

data['embedding'] = data['combined'].apply(lambda x: get_embedding(x, engine="text-embedding-ada-002"))

movie_ids = data['movie_id'].tolist() 
embeddings = data['embedding'].tolist()
vectors = list(zip(movie_ids, embeddings))


import pinecone

pinecone.init(api_key="", environment="")

index_name = "movie-recommendations"
pinecone.create_index(index_name, dimension=len(embeddings[0]))

index = pinecone.Index(index_name)
index.upsert(vectors)


from transformers import LlamaForCausalLM, LlamaTokenizer

model_name = "meta-llama/Llama-2-7b-chat-hf"
model = LlamaForCausalLM.from_pretrained(model_name)
tokenizer = LlamaTokenizer.from_pretrained(model_name)


user_query = ""
query_embedding = get_embedding(user_query, engine="text-embedding-ada-002")
search_results = index.query(query_embedding, top_k=5, include_values=True)

# Prepare the context for Llama 2 based on search results
context = ""
for match in search_results['matches']:
    movie_info = data.loc[data['movie_id'] == match['id'], 'combined'].values[0]
    context += f"{movie_info}\n"



prompt = f"Given the user's preference: '{user_query}', and considering the following movies:\n{context}\nWhat movie would you recommend?"

inputs = tokenizer(prompt, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=100)
recommendation = tokenizer.decode(output[0], skip_special_tokens=True)

print(recommendation)



def rate_movie(movie_title, user_rating):
    if movie_title in data['movie_title'].values:
        data.loc[data['movie_title'] == movie_title, 'user_rating'] = user_rating
        data.loc[data['movie_title'] == movie_title, 'rating_count'] += 1

        return data.loc[data['movie_title'] == movie_title, 'rating_count'].values[0]
        
    else:
        print(f"Movie '{movie_title}' not found in the dataset.")
        return None

# Example usage:
movie_title = ""
int user_rating
rating_count = rate_movie(movie_title, user_rating)

def update_embedding_after_30_ratings(movie_title):
    rating_count = data.loc[data['movie_title'] == movie_title, 'rating_count'].values[0]
    if rating_count % 30 == 0:
        new_embedding = get_embedding(data.loc[data['movie_title'] == movie_title, 'combined'].values[0], engine="text-embedding-ada-002")

        movie_id = data.loc[data['movie_title'] == movie_title, 'movie_id'].values[0]
        index.upsert([(movie_id, new_embedding)])

        print(f"Updated embedding for '{movie_title}' after {rating_count} ratings.")
    else:
        print(f"Embedding not updated yet. {rating_count} ratings received for '{movie_title}'.")

update_embedding_after_30_ratings(movie_title)



def recommend_and_rate_movie(user_query, movie_title, user_rating):
    query_embedding = get_embedding(user_query, engine="text-embedding-ada-002")
    search_results = index.query(query_embedding, top_k=5, include_values=True)
    
    context = ""
    for match in search_results['matches']:
        movie_info = data.loc[data['movie_id'] == match['id'], 'combined'].values[0]
        context += f"{movie_info}\n"

    prompt = f"Given the user's preference: '{user_query}', and considering the following movies:\n{context}\nWhat movie would you recommend?"
    inputs = tokenizer(prompt, return_tensors="pt")
    output = model.generate(**inputs, max_new_tokens=100)
    recommendation = tokenizer.decode(output[0], skip_special_tokens=True)
    
    rating_count = rate_movie(movie_title, user_rating)

    update_embedding_after_30_ratings(movie_title)

    return recommendation

# Example usage:
user_query = ""
movie_title = ""
user_rating 
recommendation = recommend_and_rate_movie(user_query, movie_title, user_rating)
print("Recommendation:", recommendation)
