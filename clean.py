#!/Library/Frameworks/Python.framework/Versions/3.8/bin/python3

import pandas as pd
import numpy as np

import warnings
from pandas.core.common import SettingWithCopyWarning

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

# Import Datasets
pathname = '/Users/xavierthomas/Desktop/the-movies-dataset'
ratings = pd.read_csv(pathname + '/' + 'ratings_small.csv', low_memory=False)
credits = pd.read_csv(pathname + '/' + 'credits.csv', low_memory=False)
keywords = pd.read_csv(pathname + '/' + 'keywords.csv', low_memory=False)
links = pd.read_csv(pathname + '/' + 'links.csv', low_memory=False)
metadata = pd.read_csv(pathname + '/' + 'movies_metadata.csv', low_memory=False)

n_actors = 6

# To calculate rating
m = 434
C = 5.61


def assign_rating(x):
    v = metadata[metadata['title'] == x]['vote_count']
    R = metadata[metadata['title'] == x]['vote_average']
    rating = v * R / (v + m) + m * C / (m + v)
    return rating


metadata['rating'] = assign_rating(metadata['title'])  # calculating ratings for every title

df = metadata[['title', 'genres', 'id', 'popularity',
               'rating']]  # Only taking Columns that contribute most to the content of the Movie

df_rating = df.sort_values('rating', ascending=False)  # sort titles according to ratings

df = df_rating.head(800)  # To get the best 800 movies according to rating

# converting 'id' to type int
keywords['id'] = keywords['id'].astype('int')
credits['id'] = credits['id'].astype('int')
df['id'] = df['id'].astype('int')

# merging credits and keywords to dataframe df
df = df.merge(credits, on='id')
df = df.merge(keywords, on='id')

# cleaning up genres,keywords,actors columns
from ast import literal_eval

df['genres'] = df['genres'].fillna('[]').apply(literal_eval).apply(
    lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
df['keywords'] = df['keywords'].fillna('[]').apply(literal_eval).apply(
    lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
df['actors'] = df['cast'].fillna('[]').apply(literal_eval).apply(
    lambda x: [i['name'] for i in x] if isinstance(x, list) else []).apply(
    lambda x: x[0:n_actors])  # taking top n actors

# cleaning up cast and crew columns
df['cast'] = df['cast'].apply(literal_eval)
df['crew'] = df['crew'].apply(literal_eval)


# To remove documentaries from the dataframe

def check_doc(x):
    if 'Documentary' in x:
        return True
    else:
        return False


df = df.drop(df[(df.genres.apply(check_doc))].index)


# to get the Director Name

def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan


df['director'] = df['crew'].apply(get_director)

del df['cast']
del df['crew']

# converting actor names to lower case and joining first and last names
df['actors'] = df['actors'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])

# converting director name to lower case and joining first and last names
df['director'] = df["director"].str.replace(" ", "")
df['director'] = df['director'].str.lower()
df['director'] = df['director'].apply(lambda x: [x])

# dropping NA values
df2 = df.dropna(axis=0, subset=['title'])

# print(df2.head())
