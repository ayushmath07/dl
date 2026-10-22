import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
def prep(ds, n):
    def f(ex):
        img = tf.image.resize(ex["image"], (128, 128)) / 255.
        m = tf.cast(tf.image.resize(ex["segmentation_mask"], (128, 128), method="nearest") - 1, tf.int32)
        return img, m
    return ds.map(f).batch(16).take(n)
tr = prep(tfds.load("oxford_iiit_pet", split="train"), 150)
te = prep(tfds.load("oxford_iiit_pet", split="test"), 25)
def block(x, f):
    x = keras.layers.Conv2D(f, 3, padding="same", activation="relu")(x)
    return keras.layers.Conv2D(f, 3, padding="same", activation="relu")(x)
inp = keras.Input((128, 128, 3))
c1 = block(inp, 32)
c2 = block(keras.layers.MaxPooling2D()(c1), 64)
c3 = block(keras.layers.MaxPooling2D()(c2), 128)
c4 = block(keras.layers.Concatenate()([keras.layers.UpSampling2D()(c3), c2]), 64)
c5 = block(keras.layers.Concatenate()([keras.layers.UpSampling2D()(c4), c1]), 32)
m = keras.Model(inp, keras.layers.Conv2D(3, 1, activation="softmax")(c5))
m.compile("adam", "sparse_categorical_crossentropy")
m.fit(tr, epochs=3, verbose=0)
for img, mask in te.take(1):
    p = np.argmax(m.predict(img, verbose=0), axis=-1)
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(img[0])
    ax[1].imshow(mask[0, :, :, 0])
    ax[2].imshow(p[0])
    plt.show()
inter, union = np.zeros(3), np.zeros(3)
for img, mask in te:
    p = np.argmax(m.predict(img, verbose=0), axis=-1)
    t = mask[:, :, :, 0].numpy()
    for c in range(3):
        inter[c] += np.logical_and(p == c, t == c).sum()
        union[c] += np.logical_or(p == c, t == c).sum()
print("Mean IoU:", round(float(np.mean(inter / np.maximum(union, 1))), 4))
# for img, mask in te.take(1):
#     p = np.argmax(m.predict(img, verbose=0), axis=-1)
#     fig, ax = plt.subplots(1, 2, figsize=(8, 4))
#     ax[0].imshow(img[0]); ax[0].axis("off")
#     ax[1].imshow(p[0]); ax[1].axis("off")
#     plt.show()
