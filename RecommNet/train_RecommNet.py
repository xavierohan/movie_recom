import tensorflow as tf
import keras
from RecommNet.model_RecommNet import model
from RecommNet import d

model.compile(keras.optimizers.Adam(learning_rate=0.01),
              loss='MSE',
              metrics=['mse', 'mae', 'mape'])

history = model.fit(
    [d.userId, d.id, d.genre1, d.genre2, d.key1, d.key2],
    d.rating,
    batch_size=2000,
    epochs=2,
    verbose=0,
    validation_split=.1,
)
