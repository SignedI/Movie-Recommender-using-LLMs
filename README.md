# Movie-Recommender-using-LLMs

In this Project, I build a Content-based Movie Recommender System, incorporating the use of LLMs. I use Llama 2 as the Language model and Pinecone for the Vector Database. LangChain was used to manage the entire workflow, ensuring seamless integration between the language model and the vector database. Semantic search was also incorporated to update the Vector space embeddings after a given amount of time has passed or after the movie has been reviewed by a given number of people. This ensures that the VectorDB evolves with time based on all of the users' changing reviews, allowing for more user-friendly searches.

This RecSys was built on the Netflix movies and TV-shows dataset which is available on Kaggle as well. OMDb API was used to access the IMDb ratings and vote-counts for the different movies, making them a part of the dataset for better user-friendly recommendations.

