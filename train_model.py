import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import pickle

ratings = pd.read_csv("ratings.csv")

matrix = ratings.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)

similarity = cosine_similarity(matrix)

pickle.dump(similarity, open("model.pkl", "wb"))

print("Recommendation model trained successfully!")
