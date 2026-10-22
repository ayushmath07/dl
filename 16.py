import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
zp = tf.keras.utils.get_file('jena_climate_2009_2016.csv.zip', 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip', extract=True)
t = pd.read_csv(os.path.splitext(zp)[0])['T (degC)'].values[:60000].astype('float64')
mu, sd = t[:48000].mean(), t[:48000].std()
t = (t - mu) / sd
W = 48
X = np.array([t[i:i + W] for i in range(len(t) - W)])[..., None]
y = t[W:]
n = 48000
m = tf.keras.Sequential([layers.LSTM(32, input_shape=(W, 1)), layers.Dense(1)])
m.compile('adam', 'mse')
m.fit(X[:n], y[:n], epochs=3, batch_size=128, verbose=0)
p = m.predict(X[n:], verbose=0).ravel() * sd + mu
a = y[n:] * sd + mu
print('MAE:', np.mean(np.abs(a - p)))
print('RMSE:', np.sqrt(np.mean((a - p) ** 2)))
plt.plot(a[:300], label='Actual')
plt.plot(p[:300], label='Predicted')
plt.legend()
plt.show()
# print('First test point actual temperature:', round(float(a[0]), 2), 'degC')
# print('First test point predicted temperature:', round(float(p[0]), 2), 'degC')
# print('Absolute error:', round(abs(float(a[0] - p[0])), 2), 'degC')
