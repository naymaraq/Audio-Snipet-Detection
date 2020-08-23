from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np

patterns =  np.load('tracks/snipets/patterns.npy', allow_pickle=True)
snipets =  np.load('tracks/snipets/snipets.npy', allow_pickle=True)

patterns = patterns.reshape(-1, patterns.shape[1]*patterns.shape[2])
snipets = snipets.reshape(-1, snipets.shape[1]*snipets.shape[2])

X = np.concatenate((patterns, snipets), axis=1)
y =  np.load('tracks/snipets/labels.npy', allow_pickle=True)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

clf = LogisticRegression(random_state=0).fit(X_train, y_train)
print(clf.score(X_test, y_test))