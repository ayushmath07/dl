import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
(x, _), _ = tf.keras.datasets.fashion_mnist.load_data()
x = ((x.astype('float32') - 127.5) / 127.5)[..., None]
ds = tf.data.Dataset.from_tensor_slices(x).shuffle(60000).batch(256)
G = tf.keras.Sequential([layers.Dense(12544, input_shape=(100,)), layers.LeakyReLU(0.2), layers.Reshape((7, 7, 256)), layers.Conv2DTranspose(128, 5, 2, 'same'), layers.LeakyReLU(0.2), layers.Conv2DTranspose(64, 5, 2, 'same'), layers.LeakyReLU(0.2), layers.Conv2D(1, 5, 'same', activation='tanh')])
D = tf.keras.Sequential([layers.Conv2D(64, 5, 2, 'same', input_shape=(28, 28, 1)), layers.LeakyReLU(0.2), layers.Conv2D(128, 5, 2, 'same'), layers.LeakyReLU(0.2), layers.Flatten(), layers.Dense(1)])
go, do = tf.keras.optimizers.Adam(2e-4), tf.keras.optimizers.Adam(2e-4)
bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
seed = tf.random.normal((16, 100))
def samples(ep):
    im = G(seed, training=False).numpy()
    _, ax = plt.subplots(4, 4, figsize=(4, 4))
    for i, a in enumerate(ax.flat):
        a.imshow(im[i, :, :, 0] * 0.5 + 0.5, cmap='gray')
        a.axis('off')
    plt.suptitle(f'Epoch {ep}')
    plt.show()
dl, gl = [], []
for ep in range(30):
    dt = gt = n = 0
    for real in ds:
        noise = tf.random.normal((tf.shape(real)[0], 100))
        with tf.GradientTape() as t1, tf.GradientTape() as t2:
            fake = G(noise, training=True)
            dr, dfl = D(real, training=True), D(fake, training=True)
            dloss = bce(tf.ones_like(dr), dr) + bce(tf.zeros_like(dfl), dfl)
            gloss = bce(tf.ones_like(dfl), dfl)
        do.apply_gradients(zip(t1.gradient(dloss, D.trainable_variables), D.trainable_variables))
        go.apply_gradients(zip(t2.gradient(gloss, G.trainable_variables), G.trainable_variables))
        dt += dloss.numpy()
        gt += gloss.numpy()
        n += 1
    dl.append(dt / n)
    gl.append(gt / n)
    print(ep + 1, round(dl[-1], 3), round(gl[-1], 3))
    if (ep + 1) % 5 == 0:
        samples(ep + 1)
plt.plot(dl, label='Discriminator loss')
plt.plot(gl, label='Generator loss')
plt.legend()
plt.show()
