import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
X, y = load_wine(return_X_y=True)
X = StandardScaler().fit_transform(X)
xtr, xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=1, stratify=y)
m = tf.keras.Sequential([layers.Dense(32, activation='relu', input_shape=(13,)), layers.Dense(16, activation='relu'), layers.Dense(3, activation='softmax')])
m.compile('adam', 'sparse_categorical_crossentropy', metrics=['accuracy'])
m.fit(xtr, ytr, epochs=60, batch_size=8, verbose=0)
p = np.argmax(m.predict(xte, verbose=0), axis=1)
print('Accuracy:', accuracy_score(yte, p))
print('Precision:', precision_score(yte, p, average='macro'))
print('Recall:', recall_score(yte, p, average='macro'))
print('Confusion matrix:')
print(confusion_matrix(yte, p))
# print('Sample features:', xte[0])
# print('Predicted class:', int(p[0]))
# print('Actual class:', int(yte[0]))
