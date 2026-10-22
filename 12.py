import time
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers
def xy(ds):
    X, Y = [], []
    for t, l in ds:
        X.append(t.numpy().decode())
        Y.append(l.numpy())
    return X, np.array(Y)
xtr, ytr = xy(tfds.load('ag_news_subset', split='train[:20000]', as_supervised=True))
xte, yte = xy(tfds.load('ag_news_subset', split='test[:2000]', as_supervised=True))
vec = layers.TextVectorization(max_tokens=10000, output_sequence_length=100)
vec.adapt(xtr)
V = len(vec.get_vocabulary())
Xtr, Xte = vec(xtr).numpy(), vec(xte).numpy()
class TPE(layers.Layer):
    def __init__(self, d):
        super().__init__()
        self.t = layers.Embedding(V, d)
        self.p = layers.Embedding(100, d)
    def call(self, x):
        return self.t(x) + self.p(tf.range(tf.shape(x)[1])[None, :])
def transformer():
    i = tf.keras.Input((100,))
    h = TPE(64)(i)
    a = layers.MultiHeadAttention(4, 64)(h, h)
    h = layers.LayerNormalization()(h + a)
    f = layers.Dense(64)(layers.Dense(128, activation='relu')(h))
    h = layers.LayerNormalization()(h + f)
    o = layers.Dense(4, activation='softmax')(layers.GlobalAveragePooling1D()(h))
    return tf.keras.Model(i, o)
def lstm_model():
    return tf.keras.Sequential([layers.Embedding(V, 64), layers.LSTM(64), layers.Dense(4, activation='softmax')])
for name, model in [('Transformer', transformer()), ('LSTM', lstm_model())]:
    model.compile('adam', 'sparse_categorical_crossentropy', metrics=['accuracy'])
    t0 = time.time()
    model.fit(Xtr, ytr, epochs=3, batch_size=64, verbose=0)
    print(name, 'accuracy:', round(model.evaluate(Xte, yte, verbose=0)[1], 4), 'train time:', round(time.time() - t0, 1), 's')
# print('Text:', xte[0][:100])
# print('Predicted class:', int(np.argmax(model.predict(Xte[:1], verbose=0)[0])))
# print('Actual class:', int(yte[0]))
