import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

df = pd.read_csv("https://huggingface.co/datasets/DanielKiani/RetailRocket-Recommender-Data/resolve/main/data/RetailRocket-Recommender-Data/data/events.csv")
df = df[df.event.isin(["view", "addtocart", "transaction"])][["visitorid", "itemid"]].drop_duplicates()
uc = df.visitorid.value_counts()
ic = df.itemid.value_counts()
df = df[df.visitorid.isin(uc[uc >= 10].index) & df.itemid.isin(ic[ic >= 10].index)]
umap = {u: i for i, u in enumerate(df.visitorid.unique())}
imap = {t: i for i, t in enumerate(df.itemid.unique())}
df["u"] = df.visitorid.map(umap)
df["i"] = df.itemid.map(imap)
nu, ni = len(umap), len(imap)
df = df.sample(frac=1, random_state=42).reset_index(drop=True).head(200000)
cut = int(0.8 * len(df))
train, test = df.iloc[:cut], df.iloc[cut:]

pu, pi = train.u.values, train.i.values
neg = np.random.randint(0, ni, len(pu))
U = np.concatenate([pu, pu])
I = np.concatenate([pi, neg])
y = np.array([1.0] * len(pu) + [0.0] * len(pu))

uin = keras.Input(shape=(1,))
iin = keras.Input(shape=(1,))
u_emb = keras.layers.Embedding(nu, 32)
i_emb = keras.layers.Embedding(ni, 32)
ue = keras.layers.Flatten()(u_emb(uin))
ie = keras.layers.Flatten()(i_emb(iin))
out = keras.layers.Dense(1, activation="sigmoid")(keras.layers.Dot(axes=1)([ue, ie]))
model = keras.Model([uin, iin], out)
model.compile(optimizer="adam", loss="binary_crossentropy")
model.fit([U, I], y, epochs=3, batch_size=256, verbose=0)

Uw, Iw = u_emb.get_weights()[0], i_emb.get_weights()[0]
seen = train.groupby("u")["i"].apply(set).to_dict()
truth = test.groupby("u")["i"].apply(set).to_dict()
recalls = []
for u, tset in list(truth.items())[:200]:
    scores = Uw[u] @ Iw.T
    for t in seen.get(u, ()):
        scores[t] = -1e9
    topk = np.argsort(scores)[-10:]
    recalls.append(len(set(topk.tolist()) & tset) / len(tset))
print("Recall@10:", round(float(np.mean(recalls)), 4))
