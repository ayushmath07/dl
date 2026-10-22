import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
def prep(split, n):
    ds = tfds.load("cats_vs_dogs", split=split, as_supervised=True)
    return ds.map(lambda x, y: (tf.image.resize(x, (128, 128)) / 255., tf.cast(y, tf.float32))).shuffle(1000).batch(32).take(n)
tr, te = prep("train[:80%]", 120), prep("train[80%:]", 30)
aug = keras.Sequential([keras.layers.RandomFlip("horizontal"), keras.layers.RandomRotation(0.2), keras.layers.RandomZoom(0.2), keras.layers.RandomTranslation(0.1, 0.1)])
def build(use_aug):
    layers = [keras.layers.Input((128, 128, 3))]
    if use_aug:
        layers.append(aug)
    layers += [keras.layers.Conv2D(32, 3, activation="relu"), keras.layers.MaxPooling2D(), keras.layers.Conv2D(64, 3, activation="relu"), keras.layers.MaxPooling2D(), keras.layers.Flatten(), keras.layers.Dense(64, activation="relu"), keras.layers.Dense(1, activation="sigmoid")]
    m = keras.Sequential(layers)
    m.compile("adam", "binary_crossentropy", ["accuracy"])
    return m
for name, use in [("Without augmentation", False), ("With augmentation", True)]:
    m = build(use)
    h = m.fit(tr, validation_data=te, epochs=5, verbose=0)
    print(name, "| train acc:", round(h.history["accuracy"][-1], 4), "| val acc:", round(h.history["val_accuracy"][-1], 4))
# import matplotlib.pyplot as plt
# xb, yb = next(iter(te.take(1)))
# p = float(m.predict(xb[:1], verbose=0)[0, 0])
# plt.imshow(xb[0])
# plt.title(f"pred: {'dog' if p > 0.5 else 'cat'} ({p:.2f}) | true: {'dog' if int(yb[0]) == 1 else 'cat'}")
# plt.axis("off")
# plt.show()
