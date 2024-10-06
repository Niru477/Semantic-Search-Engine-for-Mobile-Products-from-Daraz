# Semantic-Search-Engine-for-Mobile-Products-from-Daraz
# Project Overview
This project implements a semantic search engine that retrieves similar mobile product titles based on user queries. Using natural language processing (NLP) techniques and pre-trained transformer models, the system provides relevant product information such as price, rating, and availability.

#Workflow
1. Data Preprocessing and Embedding Storage
The dataset (product titles and details) is passed through a preprocessing pipeline to clean and standardize the data.
After preprocessing, the product titles are converted into vector representations (document embeddings) using a SentenceTransformer model.
These document embeddings are stored for efficient retrieval.
2. Query Processing and Embedding Generation
When a user inputs a search query, it is passed through the same preprocessing pipeline to clean and normalize the query text.
The cleaned query is then converted into vector embeddings (query embeddings) using the same SentenceTransformer model.
3. Similarity Computation
The query embeddings are compared with the stored document embeddings using cosine similarity to find the closest matches.
The most similar documents (product titles) are retrieved and ranked based on their similarity scores.
4. Results
The top results, including relevant product details (title, price, rating, etc.), are returned to the user with clickable links to the product pages.
