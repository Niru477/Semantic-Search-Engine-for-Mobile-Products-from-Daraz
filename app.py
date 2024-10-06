from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging
import os

# Initialize the Flask application
app = Flask(__name__)
CORS(app)  # Enable CORS

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load the pre-trained BERT model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Load dataset with error handling
def load_data():
    try:
        df = pd.read_csv('D:\\Semantic_Search_Engine\\cleaned_dataset.csv')  
        logging.info("Dataset Loaded Successfully.")
        return df
    except FileNotFoundError:
        logging.error("The specified file was not found.")
        return None
    except pd.errors.EmptyDataError:
        logging.error("The file is empty.")
        return None
    except pd.errors.ParserError:
        logging.error("Error parsing the file.")
        return None

df = load_data()
if df is None:
    exit()

titles = df['title'].values  

# Check if title_embeddings.npy exists before loading
embeddings_file_path = 'title_embeddings.npy'
if not os.path.exists(embeddings_file_path):
    logging.error("Title embeddings file not found.")
    exit()

title_embeddings = np.load(embeddings_file_path)

# @app.route('/search', methods=['POST'])
# def search():
#     data = request.get_json()  
#     query = data.get('query', '').strip()  # Get query and strip whitespace

#     if not query or len(query) < 3:
#         return jsonify({'error': 'Query must be at least 3 characters long.'}), 400

#     try:
#         # Encode the query
#         query_embedding = model.encode(query)

#         # Calculate cosine similarities
#         similarities = cosine_similarity([query_embedding], title_embeddings)[0]

#         # Get the top 5 results
#         top_n = 5
#         top_indices = np.argsort(similarities)[-top_n:][::-1]  # Indices of top matches

#         # Create a response with the titles, their corresponding similarity scores, and links
#         results = [
#             {
#                 'title': titles[i], 
#                 'similarity': float(similarities[i]),
#                 'link': df['full_link'].values[i]  # Add the link here
#             } 
#             for i in top_indices
#         ]

#         return jsonify(results)  # Return results as JSON

#     except Exception as e:
#         logging.error(f"An error occurred during search: {str(e)}")
#         return jsonify({'error': 'An internal error occurred. Please try again later.'}), 500

# @app.route('/search', methods=['POST'])
# def search():
#     data = request.get_json()  
#     query = data.get('query', '').strip()  # Get query and strip whitespace

#     if not query or len(query) < 3:
#         return jsonify({'error': 'Query must be at least 3 characters long.'}), 400

#     # Define mobile-related keywords including earphones and accessories
#     mobile_keywords = [
#     # General Terms
#     'mobile', 'phone', 'smartphone', 'android', 'ios',
#     'tablet', 'device', 'cellular', 'gadget',
#     'earphones', 'headphones', 'charging', 'wireless',
#     'Bluetooth', 'accessory', 'charger', 'cable',
    
#     # Brands
#     'Samsung', 'Apple', 'Xiaomi', 'OnePlus', 
#     'Nokia', 'Huawei', 'Sony', 'LG', 
#     'Oppo', 'Vivo', 'Motorola', 'Google', 
#     'Realme', 'HTC', 'Lenovo', 'Asus', 
#     'ZTE', 'Poco', 'Honor', 'BlackBerry',
    
#     # Features
#     'camera', 'display', 'battery', 'processor', 
#     'RAM', 'storage', 'fingerprint', 'face recognition',
#     'waterproof', 'durable', 'lightweight', 'compact',
#     'dual SIM', '4G', '5G', 'WiFi', 'NFC',
    
#     # Accessories
#     'screen protector', 'case', 'cover', 'pop socket',
#     'tripod', 'selfie stick', 'car mount', 'docking station',
    
#     # Technologies
#     'GPS', 'Bluetooth', 'LTE', 'WiFi', 
#     'HD', 'OLED', 'AMOLED', 'LCD', 'retina',
    
#     # Types of Phones
#     'flagship', 'mid-range', 'budget', 'feature phone',
#     'refurbished', 'unlocked', 'contract phone', 'prepaid phone',
    
#     # Related Terms
#     'charger', 'power bank', 'fast charging', 'wireless charging',
#     'earbuds', 'smartwatch', 'fitness tracker', 'VR headset',
    
#     # Usage Contexts
#     'gaming', 'photography', 'business', 'social media',
#     'streaming', 'video calls', 'music', 'internet browsing',
    
#     # Others
#     'buy', 'discount', 'sale', 'offer', 'best price', 
#     'reviews', 'comparison', 'new', 'latest',
# ]

#     # Check if the query contains any of the keywords
#     if not any(keyword in query.lower() for keyword in mobile_keywords):
#         return jsonify({'error': 'Please enter a query related to mobile phones or accessories.'}), 400

#     try:
#         # Encode the query
#         query_embedding = model.encode(query)

#         # Calculate cosine similarities
#         similarities = cosine_similarity([query_embedding], title_embeddings)[0]

#         # Get the top 5 results
#         top_n = 5
#         top_indices = np.argsort(similarities)[-top_n:][::-1]  # Indices of top matches

#         # Create a response with the titles, their corresponding similarity scores, and links
#         results = [
#             {
#                 'title': titles[i], 
#                 'similarity': float(similarities[i]),
#                 'link': df['full_link'].values[i]  # Add the link here
#             } 
#             for i in top_indices
#         ]

#         return jsonify(results)  # Return results as JSON

#     except Exception as e:
#         logging.error(f"An error occurred during search: {str(e)}")
#         return jsonify({'error': 'An internal error occurred. Please try again later.'}), 500


# @app.route('/search', methods=['POST'])
# def search():
#     data = request.get_json()  
#     query = data.get('query', '').strip()  # Get query and strip whitespace

#     # Log received query
#     logging.info(f"Received query: '{query}'")

#     if not query or len(query) < 3:
#         return jsonify({'error': 'Query must be at least 3 characters long.'}), 400

#     # Define mobile-related keywords including earphones and accessories
#     mobile_keywords = [
#         # General Terms
#         'mobile', 'phone', 'smartphone', 'android', 'ios',
#         'tablet', 'device', 'cellular', 'gadget',
#         'earphones', 'headphones', 'charging', 'wireless',
#         'Bluetooth', 'accessory', 'charger', 'cable',

#         # Brands
#         'Samsung', 'Apple', 'Xiaomi', 'OnePlus', 
#         'Nokia', 'Huawei', 'Sony', 'LG', 
#         'Oppo', 'Vivo', 'Motorola', 'Google', 
#         'Realme', 'HTC', 'Lenovo', 'Asus', 
#         'ZTE', 'Poco', 'Honor', 'BlackBerry',

#         # Features
#         'camera', 'display', 'battery', 'processor', 
#         'RAM', 'storage', 'fingerprint', 'face recognition',
#         'waterproof', 'durable', 'lightweight', 'compact',
#         'dual SIM', '4G', '5G', 'WiFi', 'NFC',

#         # Accessories
#         'screen protector', 'case', 'cover', 'pop socket',
#         'tripod', 'selfie stick', 'car mount', 'docking station',

#         # Technologies
#         'GPS', 'Bluetooth', 'LTE', 'WiFi', 
#         'HD', 'OLED', 'AMOLED', 'LCD', 'retina',

#         # Types of Phones
#         'flagship', 'mid-range', 'budget', 'feature phone',
#         'refurbished', 'unlocked', 'contract phone', 'prepaid phone',

#         # Related Terms
#         'charger', 'power bank', 'fast charging', 'wireless charging',
#         'earbuds', 'smartwatch', 'fitness tracker', 'VR headset',

#         # Usage Contexts
#         'gaming', 'photography', 'business', 'social media',
#         'streaming', 'video calls', 'music', 'internet browsing',

#         # Others
#         'buy', 'discount', 'sale', 'offer', 'best price', 
#         'reviews', 'comparison', 'new', 'latest',
#     ]

#     # Check if the query contains any of the keywords
#     contains_keyword = any(keyword in query.lower() for keyword in mobile_keywords)
#     logging.info(f"Contains keyword: {contains_keyword}")
    
#     if not contains_keyword:
#         return jsonify({'error': 'Please enter a query related to mobile phones or accessories.'}), 400

#     try:
#         # Encode the query
#         query_embedding = model.encode(query)

#         # Calculate cosine similarities
#         similarities = cosine_similarity([query_embedding], title_embeddings)[0]

#         # Get the top 5 results
#         top_n = 5
#         top_indices = np.argsort(similarities)[-top_n:][::-1]  # Indices of top matches

#         # Create a response with the titles, their corresponding similarity scores, and links
#         results = [
#             {
#                 'title': titles[i], 
#                 'similarity': float(similarities[i]),
#                 'link': df['full_link'].values[i]  
#             } 
#             for i in top_indices
#         ]

#         return jsonify(results)  # Return results as JSON

#     except Exception as e:
#         logging.error(f"An error occurred during search: {str(e)}")
#         return jsonify({'error': 'An internal error occurred. Please try again later.'}), 500

@app.route('/search', methods=['POST'])
def search():
    data = request.get_json()  
    query = data.get('query', '').strip()  # Get query and strip whitespace

    if not query or len(query) < 3:
        return jsonify({'error': 'Query must be at least 3 characters long.'}), 400

    try:
        # Encode the query
        query_embedding = model.encode(query)

        # Calculate cosine similarities
        similarities = cosine_similarity([query_embedding], title_embeddings)[0]

        # Get the top 5 results
        top_n = 5
        top_indices = np.argsort(similarities)[-top_n:][::-1]  # Indices of top matches

        # Check if the highest similarity score is below the threshold
        if similarities[top_indices[0]] < 0.45:
            return jsonify({'error': 'No relevant results found for the given query.'}), 404

        # Create a response with the titles, their corresponding similarity scores, and links
        results = [
            {
                'title': titles[i], 
                'similarity': float(similarities[i]),
                'link': df['full_link'].values[i]  # Add the link here
            } 
            for i in top_indices
        ]

        return jsonify(results)  # Return results as JSON

    except Exception as e:
        logging.error(f"An error occurred during search: {str(e)}")
        return jsonify({'error': 'An internal error occurred. Please try again later.'}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

if __name__ == '__main__':
    app.run(debug=True)
