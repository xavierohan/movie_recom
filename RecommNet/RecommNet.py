import pandas as pd
import numpy as np

pathname = '/Users/xavierthomas/Desktop/the-movies-dataset'
df_movies = pd.read_csv(pathname + '/' + 'df_large.csv')
df_ratings = pd.read_csv(pathname + '/' + 'd_large.csv')
ratings = pd.read_csv(pathname + '/' + 'ratings_small.csv')

del df_movies['overview']
# df_movies.head()

df_ratings.rename(columns = {'id':'MovieId'}, inplace = True)
df_ratings = df_ratings[['userId', 'MovieId', 'title', 'genres', 'keywords', 'rating']]

d = df_ratings

# print("Size of ratings dataframe: ",len(ratings), "  Size of movies dataframe: ",len(df_movies))

from ast import literal_eval
# To return the first 3 genres
df_movies['genres'] = df_movies['genres'].apply(literal_eval).apply(lambda x : x[0:3])

del df_movies['Unnamed: 0']

# Split the genres to separate columns
df_movies[['genre1','genre2', 'genre3']] = pd.DataFrame(df_movies.genres.tolist(), index= df_movies.index)
# df_movies.head(2)


# Drop movies that do not have two genre values
n = len(df_movies)
df_movies.dropna(subset = ["genre1", "genre2"], inplace=True)

# print("Size of DataFrame after dropping movies that do not have 2 genre valus : ",len(df_movies))
print("Number of movies dropped from original dataset (because of null values) : ", n - len(df_movies))

# Map genre1 to integer values
genre1_list = np.unique(df_movies.genre1)
g1_dict = {k: int(v) for v, k in enumerate(genre1_list)}


# Map genre2 to integer values
genre2_list = np.unique(df_movies.genre2)
g2_dict = {k: int(v) for v, k in enumerate(genre2_list)}

# Replace categorical values of genre with integer values
df_movies = df_movies.replace({"genre1": g1_dict, "genre2": g2_dict})
df_movies.head(2)

ratings = ratings.rename(columns={'movieId': 'id'})

# merge ratings and df_movies based on movieid
d = pd.merge(ratings, df_movies, on ='id' )

del ratings
del d['timestamp']
del d['popularity']
del d['release_date']
del d['actors']
del d['director']


# mapping the movieid to continous targets, as there are breaks between ids
t = dict([(y,x) for x,y in enumerate(np.unique(d['id']))])
d['id'] = d['id'].map(t)


# starting userId from 0
d['userId'] = d['userId'] - 1


# Getting the most common movie keywords

import ast
temp =[]
for i in d['keywords']:
    res = ast.literal_eval(i)
    temp.extend(res)

# print(len(temp))

from collections import Counter
Counter = Counter(temp)
most_occur = Counter.most_common(1500) #1500


# Convert most common keywords to integer values

k = dict([(y[0], x) for x, y in enumerate(most_occur)])
f = []
for i in d['keywords']:
    temp = []
    for j in ast.literal_eval(i):
        if j in k.keys() and len(temp) < 3:
            temp.append(k[j])

    f.append(temp)


d['key'] = f

d['key_count'] = d['key'].apply(lambda x: len(x))
# print("Size before dropping: ",len(d))
d = d[d['key_count'] >1] # Drop movies that have less than one most common keywords
# print("Size of DataFrame after dropping : ",len(d))


# Create columns based on keyword integer value
d[['key1','key2', 'key3']] = pd.DataFrame(d.key.tolist(), index= d.index)


# Map the Keyword values to continous values starting from 0

key_list = np.unique(list(np.unique(d['key1'])) + list(np.unique(d['key2'])))

t = dict([(y,x) for x,y in enumerate(key_list)])

d = d.replace({"key1": t, "key2":t})

del d['genre3']
del d['key']
del d['key_count']
del d['key3']


print("Number of unique movies : ",len(np.unique(d['id'])),"\nNumber of unique users: ", len(np.unique(d['userId'])),
      "\nTotal Number of enteries in the DataFrame: ", len(d))





