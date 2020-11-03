from RecommNet.RecommNet import d, df_movies
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from RecommNet.embedding_search import suggest_movies_knn, suggest_users_knn, get_knn, movies_index
from RecommNet.train_RecommNet import model
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from clean import df2
import nmslib
import argparse

parser = argparse.ArgumentParser(description='Recommend Movies ')
parser.add_argument('-m', '--movie_id', type=int, required=False, help='Enter the movie id')
parser.add_argument('-u', '--user_id', type=int, required=False, help='Enter the user id')
parser.add_argument('-p', '--performance', type=int, required=False, help='1: to get model performance')

args = vars(parser.parse_args())


if args['performance'] == 1:
    X = d[['userId', 'id', 'genre1', 'genre2', 'rating', 'key1', 'key2']]
    y = d['rating']

    X['pred'] = model.predict([[X['userId']], [X['id']], [X['genre1']], [X['genre2']], [X['key1']], [X['key2']]])

    X['diff'] = abs(X['rating'] - X['pred'])

    MSE = mean_squared_error(y_true=X.rating.values, y_pred=X.pred.values)
    MAE = mean_absolute_error(y_true=X.rating.values, y_pred=X.pred.values)

    print("MEAN SQUARED ERROR : ", MSE, "\nROOT MEAN SQUARED ERROR : ", MSE ** (0.5), "\nMEAN ABSOLUTE ERROR : ", MAE)

# Recommendations for Star Wars
#movie_id = 188  # 188 --> Star Wars
movie_id = args['movie_id']
print("\n")
print("Input Movie is : ",d[d['id'] == movie_id].head(1)['title'].item(), "\n")
j = suggest_movies_knn(movie_id, 8)
print(" Recommended Movies based on Movie Embedding are : \n", list(np.unique(d[d['id'].isin(j)]['title'])), "\n")

# Recommend similar profiles
#user_id = 288
user_id = args['user_id']
j = suggest_users_knn(user_id, 5)
print(" Recommended Users based on user Embedding are : \n", list(np.unique(d[d['userId'].isin(j)]['userId']))[:10], "\n")

# Recommendations Based on User Profile #288
#user_id = 288
user_profile = d[d['userId'] == user_id]
user_profile = user_profile[['userId', 'id', 'title', 'genre1', 'genre2', 'key1', 'key2', 'rating', 'genres']]
user_profile = user_profile[user_profile['rating'] > 4]
user_profile  # User Profile of user 288

# Finding the average movie embedding to capture user interests.
emb_layer = model.get_layer('movie_embedding')
(w,) = emb_layer.get_weights()
avg_w = 0
for i in user_profile.id:
    avg_w += w[i]
avg_w = avg_w / len(user_profile)

# Recommending movies based on average movie embedding
j = get_knn(movies_index, avg_w, 5)[0]
print(" Recommended Movies based on User Profile are : \n", list(np.unique(d[d['id'].isin(j)]['title'])), "/n")
