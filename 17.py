import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
(xtr, _), (xte, _) = tf.keras.datasets.fashion_mnist.load_data()
xtr = (xtr[:30000].astype('float32') / 255.0)[..., None]
xte = (xte.astype('float32') / 255.0)[..., None]
ntr = np.clip(xtr + 0.5 * np.random.normal(size=xtr.shape), 0, 1).astype('float32')
nte = np.clip(xte + 0.5 * np.random.normal(size=xte.shape), 0, 1).astype('float32')
i = tf.keras.Input((28, 28, 1))
h = layers.Conv2D(32, 3, 2, 'same', activation='relu')(i)
h = layers.Conv2D(64, 3, 2, 'same', activation='relu')(h)
h = layers.Conv2DTranspose(32, 3, 2, 'same', activation='relu')(h)
o = layers.Conv2DTranspose(1, 3, 2, 'same', activation='sigmoid')(h)
m = tf.keras.Model(i, o)
m.compile('adam', 'mse')
m.fit(ntr, xtr, epochs=5, batch_size=128, verbose=0)
r = m.predict(nte, verbose=0)
mse = np.mean((xte - r) ** 2)
print('MSE:', mse)
print('PSNR:', 10 * np.log10(1 / mse))
_, ax = plt.subplots(2, 10, figsize=(12, 3))
for j in range(10):
    ax[0, j].imshow(nte[j, :, :, 0], cmap='gray')
    ax[0, j].axis('off')
    ax[1, j].imshow(r[j, :, :, 0], cmap='gray')
    ax[1, j].axis('off')
plt.show()
# _, ax = plt.subplots(1, 2)
# ax[0].imshow(nte[0, :, :, 0], cmap='gray')
# ax[0].set_title('Noisy')
# ax[0].axis('off')
# ax[1].imshow(r[0, :, :, 0], cmap='gray')
# ax[1].set_title('Restored')
# ax[1].axis('off')
# plt.show()
