import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
(x, y), (xt, yt) = keras.datasets.mnist.load_data()
x = (x.reshape(-1, 784) / 255.).astype("float32")
xt = (xt.reshape(-1, 784) / 255.).astype("float32")
class Sampling(keras.layers.Layer):
    def call(self, args):
        m, v = args
        return m + tf.exp(0.5 * v) * tf.random.normal(tf.shape(m))
inp = keras.Input((784,))
h = keras.layers.Dense(512, activation="relu")(inp)
zm, zv = keras.layers.Dense(2)(h), keras.layers.Dense(2)(h)
enc = keras.Model(inp, [zm, zv, Sampling()([zm, zv])])
din = keras.Input((2,))
dec = keras.Model(din, keras.layers.Dense(784, activation="sigmoid")(keras.layers.Dense(512, activation="relu")(din)))
class VAE(keras.Model):
    def __init__(self, enc, dec):
        super().__init__()
        self.enc, self.dec = enc, dec
    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            mn, vr, z = self.enc(data)
            r = self.dec(z)
            rec = -tf.reduce_sum(data * tf.math.log(r + 1e-9) + (1 - data) * tf.math.log(1 - r + 1e-9), axis=1)
            kl = -0.5 * tf.reduce_sum(1 + vr - tf.square(mn) - tf.exp(vr), axis=1)
            loss = tf.reduce_mean(rec + kl)
        g = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(g, self.trainable_weights))
        return {"loss": loss}
vae = VAE(enc, dec)
vae.compile("adam")
vae.fit(x, epochs=5, batch_size=128, verbose=0)
means = enc.predict(xt, verbose=0)[0]
plt.figure(figsize=(8, 6))
plt.scatter(means[:, 0], means[:, 1], c=yt, s=2, cmap="tab10")
plt.colorbar()
plt.title("Latent space")
plt.show()
grid = np.array([[a, b] for b in np.linspace(3, -3, 12) for a in np.linspace(-3, 3, 12)], dtype="float32")
imgs = dec.predict(grid, verbose=0).reshape(144, 28, 28)
fig = np.zeros((28 * 12, 28 * 12))
for i, im in enumerate(imgs):
    fig[(i // 12) * 28:(i // 12 + 1) * 28, (i % 12) * 28:(i % 12 + 1) * 28] = im
plt.figure(figsize=(8, 8))
plt.imshow(fig, cmap="gray")
plt.axis("off")
plt.show()
# z = np.random.normal(size=(1, 2)).astype("float32")
# img = dec.predict(z, verbose=0).reshape(28, 28)
# plt.imshow(img, cmap="gray")
# plt.axis("off")
# plt.title("Generated digit from random latent point")
# plt.show()
