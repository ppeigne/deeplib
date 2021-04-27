import numpy as np
from optimizers import Optimizer_

class Perceptron():
    def __init__(self, n_input=2): # activation="sigmoid"):
        self.generate_params(n_input)
        #self.activation = activation

    def generate_params(self, n_input):
        self.params= [{}]
        self.params[0]['W'] = np.random.random((1, n_input))
        self.params[0]['b'] = np.random.random(1)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, X):
        return self._sigmoid((self.params[0]['W'] @ X) + self.params[0]['b'])

    def backward(self, X, y, y_pred):
        m = X.shape[1]
        X_ = np.concatenate((np.ones((1, m)), X), axis=0)
        grad = X_ @ (y_pred - y).T / m
        gradient = [{}]
        gradient[0]['b'] = grad[0]
        gradient[0]['W'] = grad[1:].T 
        return gradient

# p = Perceptron(4)
# o = Optimizer_(p)
# X = np.arange(4).reshape(4,1)
# print('params at start:',p.params)
# o.batch_optimization([0], X, 1000, .01, False)
# print('params at end:',p.params)

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score , f1_score
from sklearn.preprocessing import StandardScaler
import pandas as pd

data = load_iris()
x = data.data
y = data.target
x = StandardScaler().fit_transform(x)
X_train, X_test, y_train, y_test = train_test_split(x, y)
x_ = X_train.T
y_ = y_train.T

p1 = Perceptron(x_.shape[0])
print(p1.params)
o1 = Optimizer_(p1)
o1.batch_optimization(x_, y_ == 0, 100000,.1)
print(p1.params, end='\n\n')
prev1 = p1.forward(X_test.T)

p2 = Perceptron(x_.shape[0])
print(p2.params)
o2 = Optimizer_(p2)
o2.batch_optimization(x_, y_ == 1, 100000,.1)
print(p2.params, end='\n\n')
prev2 = p2.forward(X_test.T)

p3 = Perceptron(x_.shape[0])
print(p3.params)
o3 = Optimizer_(p3)
o3.batch_optimization(x_, y_ == 2, 100000,.1)
print(p3.params, end='\n\n')
prev3 = p3.forward(X_test.T)

prev = np.array(pd.DataFrame(np.concatenate((prev1, prev2, prev3), axis=0).T).idxmax(axis=1))
print(prev)


# met = precision_score(y_test == 2, prev3.T > .5) #, average = 'micro')
# print(met)

met = precision_score(y_test, prev.T, average = 'micro')
print(met)

#met = precision_score(y_test, y_test, average = 'micro')
#print(met)

