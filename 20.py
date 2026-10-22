import numpy as np
from tensorflow import keras
import tensorflow_datasets as tfds
text = "".join(ex["text"].numpy().decode() for ex in tfds.load("tiny_shakespeare", split="train"))[:300000]
chars = sorted(set(text))
c2i = {c: i for i, c in enumerate(chars)}
i2c = {i: c for c, i in c2i.items()}
data = np.array([c2i[c] for c in text])
def sample(m, n, temp):
    s, out = list(data[:n]), ""
    for _ in range(200):
        p = m.predict(np.array([s]), verbose=0)[0].astype("float64")
        p = np.exp(np.log(p + 1e-9) / temp)
        p /= p.sum()
        k = int(np.random.choice(len(chars), p=p))
        out += i2c[k]
        s = s[1:] + [k]
    return out
for seq_len, emb in [(40, 64), (80, 128)]:
    idx = range(0, len(data) - seq_len, seq_len)
    X = np.array([data[i:i + seq_len] for i in idx])[:6000]
    y = np.array([data[i + seq_len] for i in idx])[:6000]
    m = keras.Sequential([keras.layers.Input((seq_len,)), keras.layers.Embedding(len(chars), emb), keras.layers.LSTM(128), keras.layers.Dense(len(chars), activation="softmax")])
    m.compile("adam", "sparse_categorical_crossentropy")
    h = m.fit(X, y, epochs=3, batch_size=128, verbose=0)
    print("seq_len:", seq_len, "embedding:", emb, "loss:", round(h.history["loss"][-1], 4))
    for temp in (0.5, 1.0):
        print("temperature:", temp)
        print(sample(m, seq_len, temp))
