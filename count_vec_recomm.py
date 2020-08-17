# ONLY WORKS FOR column = ALL FOR NOW

from main import df2
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import argparse

parser = argparse.ArgumentParser(description='Recommend Movies based on count vectors ')
parser.add_argument('-m', '--movie', type=str, required=True, help='Enter the movie name')
parser.add_argument('-c', '--column', type=str, required=True,
                    help='Enter filtering basis (genres, keywords, actors, director, All')
# args = parser.parse_args()
args = vars(parser.parse_args())

# m = 'Shutter Island'

df_cv = df2.copy()
df_cv['All'] = df_cv['genres'].astype(str) + df_cv['keywords'].astype(str) + df_cv['actors'].astype(str) + df_cv[
    'director'].astype(str)

count = CountVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')

column = 'All'
count_matrix = count.fit_transform(df_cv[args['column']])

cosine_sim = cosine_similarity(count_matrix, count_matrix)

print('hello')


def recommendations_cv(title):
    recommended_movies = []

    # getting the index of the movie that matches the title
    idx = df_cv[df_cv['title'] == title].index[0]

    # creating a Series with the similarity scores in descending order
    score_series = pd.Series(cosine_sim[idx]).sort_values(ascending=False)

    # getting the indexes of the 10 most similar movies
    top_10_indexes = list(score_series.iloc[1:11].index)

    # appending the list with the titles of the best 10 matching movies
    for i in top_10_indexes:
        recommended_movies.append(list(df_cv.index)[i])

    return recommended_movies


# print(args["movie"])
y = recommendations_cv(args["movie"])

print(df_cv.loc[y, :]['title'])
