from layers import *
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

data = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(data.data, data.target)



architecture = [Dense(30, activation='relu'), 
                Dropout(keep_prob=.8),
                Dense(8),
                Dropout(keep_prob=.8),
                Dense(8),
                Dropout(keep_prob=.8),
                Dense(8),
                Dropout(keep_prob=.8),
                Dense(5),
                Dropout(keep_prob=.8),
                Dense(5), 
                Dropout(keep_prob=.8),
                Dense(5),  
                Dropout(keep_prob=.8),
                Dense(1, activation='sigmoid')] 

model = Network(architecture)

model.train(X_train.T, y_train.T)

y_pred = model.predict(X_test.T)

print(f1_score(y_test, y_pred.T))