import re
import zipfile
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
z = tf.keras.utils.get_file('s140.zip', 'https://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip')
zipfile.ZipFile(z).extractall('/tmp/s140')
df = pd.read_csv('/tmp/s140/training.1600000.processed.noemoticon.csv', encoding='latin-1', header=None, names=['y', 'i', 'd', 'f', 'user', 'text']).sample(20000, random_state=1)
df['y'] = (df.y == 4).astype(int)
vec = layers.TextVectorization(max_tokens=5000, output_sequence_length=50)
vec.adapt(df.text.values)
v = len(vec.get_vocabulary())
m = tf.keras.Sequential([vec, layers.Embedding(v, 64), layers.LSTM(64), layers.Dense(1, activation='sigmoid')])
m.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
X, Y = df.text.values, df.y.values
m.fit(X[:16000], Y[:16000], epochs=3, batch_size=64, verbose=0)
print('Accuracy:', m.evaluate(X[16000:], Y[16000:], verbose=0)[1])
df['p'] = m.predict(X, verbose=0).ravel()
sent = df.groupby('user').p.mean().to_dict()
G = nx.Graph()
for u, t in zip(df.user, df.text):
    for w in re.findall(r'@(\w+)', str(t)):
        G.add_edge(u, w)
G = G.subgraph([n for n, d in G.degree() if d > 1])
nx.draw(G, node_color=[sent.get(n, 0.5) for n in G.nodes], cmap='RdYlGn', vmin=0, vmax=1, node_size=25, width=0.3)
plt.show()
