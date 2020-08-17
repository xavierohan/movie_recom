from main import metadata
import argparse

parser = argparse.ArgumentParser(description='Recommend Movies based on count vectors ')
parser.add_argument('-g', '--genre', type=str, required=True, help='Enter genre')
parser.add_argument('-n', '--number', type=str, required=True, help='Enter n to return top n recommendations')
args = vars(parser.parse_args())


def genre_recomm(x):
    s = metadata['genres'].str.contains(x)
    t = metadata[s]
    print(t.sort_values('rating', ascending=False)['title'].head(args['number']))


genre_recomm(args['genre'])
