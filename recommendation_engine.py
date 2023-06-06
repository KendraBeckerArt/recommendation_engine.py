import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class RecommendationEngine:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.similarity_matrix = None

    def load_data(self, file_path):
        data = pd.read_csv(file_path)
        return data

    def preprocess_data(self, data):
        # Perform any necessary data preprocessing steps
        # such as handling missing values or encoding categorical variables
        # ...

        return data

    def build_similarity_matrix(self, data):
        item_descriptions = data['description']
        tfidf_matrix = self.vectorizer.fit_transform(item_descriptions)
        self.similarity_matrix = cosine_similarity(tfidf_matrix)

    def get_similar_items(self, item_id, top_n=5):
        item_index = item_id - 1  # Assuming item IDs start from 1
        item_similarity_scores = self.similarity_matrix[item_index]
        similar_item_indices = item_similarity_scores.argsort()[::-1][1:top_n+1]
        similar_items = similar_item_indices + 1  # Convert back to item IDs starting from 1
        return similar_items

# Example usage:
file_path = 'path/to/your/data.csv'

recommendation_engine = RecommendationEngine()

data = recommendation_engine.load_data(file_path)
preprocessed_data = recommendation_engine.preprocess_data(data)

recommendation_engine.build_similarity_matrix(preprocessed_data)

item_id = 1
top_similar_items = recommendation_engine.get_similar_items(item_id, top_n=5)
print("Top similar items for Item", item_id, ":")
print(top_similar_items)
