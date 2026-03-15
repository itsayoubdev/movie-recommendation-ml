import pickle
import pandas as pd

similarity = pickle.load(open("model.pkl","rb"))
movies = pd.read_csv("movies.csv")

user_id = 1

scores = similarity[user_id-1]

recommended_user = scores.argsort()[::-1][1]

print("Most similar user:", recommended_user+1)
