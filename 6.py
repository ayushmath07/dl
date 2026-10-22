import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

(x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

inp = keras.Input(shape=(28, 28))
x = keras.layers.Dense(128, activation="relu")(keras.layers.Flatten()(inp))
x = keras.layers.Dense(32, activation="relu")(x)
x = keras.layers.Dense(128, activation="relu")(x)
out = keras.layers.Reshape((28, 28))(keras.layers.Dense(784, activation="sigmoid")(x))
autoencoder = keras.Model(inp, out)
autoencoder.compile(optimizer="adam", loss="mse")
autoencoder.fit(x_train, x_train, epochs=5, batch_size=256, verbose=0)
recon = autoencoder.predict(x_test, verbose=0)
print("Reconstruction error (MSE):", np.mean((x_test - recon) ** 2))

plt.figure(figsize=(10, 4))
for i in range(5):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[i], cmap="gray")
    plt.axis("off")
    plt.subplot(2, 5, i + 6)
    plt.imshow(recon[i], cmap="gray")
    plt.axis("off")
plt.show()
# fig, axes = plt.subplots(1, 2)
# axes[0].imshow(x_test[0], cmap="gray")
# axes[0].set_title("Original")
# axes[1].imshow(recon[0], cmap="gray")
# axes[1].set_title("Reconstructed")
# plt.show()
# print("Sample reconstruction MSE:", float(np.mean((x_test[0] - recon[0]) ** 2)))
