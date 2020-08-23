from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import pickle

patterns =  np.load('tracks/snipets/patterns.npy', allow_pickle=True)
snipets =  np.load('tracks/snipets/snipets.npy', allow_pickle=True)
patterns = patterns.reshape(-1, patterns.shape[1]*patterns.shape[2])
snipets = snipets.reshape(-1, snipets.shape[1]*snipets.shape[2])

X = np.concatenate((patterns, snipets), axis=1)
y =  np.load('tracks/snipets/labels.npy', allow_pickle=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

best_parameters = { "tree_method": "hist",
                "colsample_bytree": 0.2,
                "learning_rate": 0.3,
                "max_depth": 10,
                "alpha": 10,
                "n_estimators": 40,

}

#xgboost
xgb_clf = xgb.XGBClassifier(**best_parameters)
xgb_clf.fit(X_train, y_train)

y_pred = xgb_clf.predict(X_test)
cmx = confusion_matrix(y_pred, y_test)
print(cmx)
print("Accuracy: ", accuracy_score(y_pred, y_test))

#save model
filename = 'models/is_snipet_in.sav'
pickle.dump(xgb_clf, open(filename, 'wb'))

