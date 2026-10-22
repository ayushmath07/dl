import time
import numpy as np
import tensorflow as tf
from tensorflow import keras

(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=10000)
x_train, y_train = x_train[:15000], y_train[:15000]
x_train = keras.utils.pad_sequences(x_train, maxlen=200)
x_test_pad = keras.utils.pad_sequences(x_test, maxlen=200)
long_idx = [i for i, s in enumerate(x_test) if len(s) > 400]
x_long = keras.utils.pad_sequences([x_test[i] for i in long_idx], maxlen=500)
y_long = np.array(y_test)[long_idx]

for name, cell in [("SimpleRNN", keras.layers.SimpleRNN), ("LSTM", keras.layers.LSTM), ("GRU", keras.layers.GRU)]:
    m = keras.Sequential([keras.layers.Embedding(10000, 64), cell(64), keras.layers.Dense(1, activation="sigmoid")])
    m.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    t = time.time()
    m.fit(x_train, y_train, epochs=2, batch_size=128, verbose=0)
    train_time = time.time() - t
    acc = m.evaluate(x_test_pad, y_test, verbose=0)[1]
    long_acc = m.evaluate(x_long, y_long, verbose=0)[1]
    print(name, "Accuracy:", round(acc, 4), "Training time (s):", round(train_time, 1), "Long-review accuracy:", round(long_acc, 4))
# pred = m.predict(x_test_pad[:1], verbose=0)
# print("First 20 tokens of the review:", x_test[0][:20])
# print("Predicted sentiment:", "Positive" if pred[0][0] > 0.5 else "Negative", "| Actual:", "Positive" if y_test[0] == 1 else "Negative")
