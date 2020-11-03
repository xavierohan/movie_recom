import keras
import numpy as np
from RecommNet import d

hidden_units = (32, 4)

m_emb_size = min(len(np.unique(d.id)) // 2, 50)
u_emb_size = min(len(np.unique(d.userId)) // 2, 50)
g1_emb_size = min(len(np.unique(d.genre1)) // 2, 50)
g2_emb_size = min(len(np.unique(d.genre2)) // 2, 50)
k1_emb_size = min(len(np.unique(d.key1)) // 2, 50)
k2_emb_size = min(len(np.unique(d.key2)) // 2, 50)

# Each instance will consist of two inputs: a single user id, and a single movie id
user_id_input = keras.Input(shape=(1,), name='user_id')
movie_id_input = keras.Input(shape=(1,), name='movie_id')

g1_id_input = keras.Input(shape=(1,), name='g1_id')
g2_id_input = keras.Input(shape=(1,), name='g2_id')

k1_id_input = keras.Input(shape=(1,), name='k1_id')
k2_id_input = keras.Input(shape=(1,), name='k2_id')

user_embedded = keras.layers.Embedding(d.userId.max() + 1, m_emb_size,
                                       input_length=1, name='user_embedding')(user_id_input)
movie_embedded = keras.layers.Embedding(d.id.max() + 1, u_emb_size,
                                        input_length=1, name='movie_embedding')(movie_id_input)

g1_embedded = keras.layers.Embedding(d.genre1.max() + 1, g1_emb_size,
                                     input_length=1, name='genre1_embedding')(g1_id_input)
g2_embedded = keras.layers.Embedding(d.genre2.max() + 1, g2_emb_size,
                                     input_length=1, name='genre2_embedding')(g2_id_input)

k1_embedded = keras.layers.Embedding(d.key1.max() + 1, k1_emb_size,
                                     input_length=1, name='key1_embedding')(k1_id_input)
k2_embedded = keras.layers.Embedding(d.key2.max() + 1, k2_emb_size,
                                     input_length=1, name='key2_embedding')(k2_id_input)

# Concatenate the embeddings (and remove the useless extra dimension)
concatenated = keras.layers.Concatenate()(
    [user_embedded, movie_embedded, g1_embedded, g2_embedded, k1_embedded, k2_embedded])
out = keras.layers.Flatten()(concatenated)


from keras import backend as K

def custom_activation(x):

    return K.sigmoid(x) * 6


# Add one or more hidden layers
for n_hidden in hidden_units:
    out = keras.layers.Dense(n_hidden, activation='relu')(out)
    out = keras.layers.Dropout(0.2)(out)
    # out = keras.layers.Dense(n_hidden, activation=custom_activation)(out)

# A single output: our predicted rating
out = keras.layers.Dense(1, activation=custom_activation, name='prediction')(out)  # 'linear'

model = keras.Model(
    inputs=[user_id_input, movie_id_input, g1_id_input, g2_id_input, k1_id_input, k2_id_input],
    outputs=out,
)
