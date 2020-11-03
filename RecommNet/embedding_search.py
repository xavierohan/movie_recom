# Searching in Embedding Space
import nmslib
from train_RecommNet import model

movies_index = nmslib.init(space='angulardist', method='hnsw')
movies_index.addDataPointBatch(model.get_layer('movie_embedding').get_weights()[0])

user_index = nmslib.init(space='angulardist', method='hnsw')
user_index.addDataPointBatch(model.get_layer('user_embedding').get_weights()[0])

M = 100
efC = 1000
efS = 1000
num_threads = 6
index_time_params = {'M': M, 'indexThreadQty': num_threads, 'efConstruction': efC, 'post': 0}
query_time_params = {'efSearch': efS}

movies_index.createIndex(index_time_params)
movies_index.setQueryTimeParams(query_time_params)

user_index.createIndex(index_time_params)
user_index.setQueryTimeParams(query_time_params)


def get_knns(index, vecs, n_neighbour):
    return zip(*index.knnQueryBatch(vecs, k=n_neighbour, num_threads=6))


def get_knn(index, vec, n_neighbour):
    return index.knnQuery(vec, k=n_neighbour)


def suggest_movies_knn(movieId, n_suggest=5):
    id = movieId
    res = get_knn(movies_index, model.get_layer("movie_embedding").get_weights()[0][movieId], n_suggest)[0]
    # return df_main[df_main.id.isin([idx2movie[i] for i in res])]
    return res


def suggest_users_knn(userId, n_suggest=5):
    i = userId
    res = get_knn(user_index, model.get_layer("user_embedding").get_weights()[0][userId], n_suggest)[0]
    # return df_main[df_main.id.isin([idx2movie[i] for i in res])]
    return res