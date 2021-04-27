import numpy as np
from optimizers import Optimizer_, MomentumOptimizer_, RMSOptimizer_, AdamOptimizer_

class LinearRegression_():
    def __init__(self, optimizer="gradient"):
        self.params = None
        self.optimizer = self._select_optimizer(optimizer)

    def _select_optimizer(self, optimizer):
        options = {
            "gradient": Optimizer_()
        #    "momentum": MomentumOptimizer_(),
        #    "rms": RMSOptimizer_(),
        #    "adam": AdamOptimizer_()
        }
        return options[optimizer]

    def _generate_params(self, n_input):
        self.params= [{}]
        self.params[0]['theta'] = np.random.random((n_input + 1, 1))

    def _add_intercept(self, x):
        return np.concatenate((np.ones((x.shape[0], 1)), x), axis= 1)

    # def forward(self, X):
    #     if not self.params:
    #         self._generate_params(X.shape[1])
    #     X_ = self._add_intercept(X)
    #     return self._sigmoid(X_ @ self.params[0]['theta'])

    def predict(self, X):
        if not self.params:
            self._generate_params(X.shape[1])
        X_ = self._add_intercept(X)
        return X_ @ self.params[0]['theta']

        #return self.forward(X)

    # def backward(self, X, y, y_pred):
    #     X_ = self._add_intercept(X)
    #     y = y.reshape(-1, 1)
    #     gradient = [{}]
    #     gradient[0]['theta'] = X_.T @ (y_pred - y) / X.shape[1]
    #     return gradient

    def gradient(self, X, y, y_pred):
        X_ = self._add_intercept(X)
        y = y.reshape(-1, 1)
        gradient = [{}]
        gradient[0]['theta'] = X_.T @ (y_pred - y) / X.shape[1]
        return gradient

    def fit(self, X, y, n_cycles=10000, learning_rate=0.1):
        self.optimizer.optimize(self, X, y, n_cycles, learning_rate)

class LogisticRegression_(LinearRegression_):
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def predict(self, X):
        if not self.params:
            self._generate_params(X.shape[1])
        X_ = self._add_intercept(X)
        return self._sigmoid(X_ @ self.params[0]['theta'])

    # def gradient(self, X, y, y_pred):
    #     X_ = self._add_intercept(X)
    #     y = y.reshape(-1, 1)
    #     gradient = [{}]
    #     gradient[0]['theta'] = X_.T @ (y_pred - y) / X.shape[1]
    #     return gradient

class RidgeRegression_(LinearRegression_):
    def gradient(self, X, y, y_pred):
        m = X.shape[1]
        X_ = self._add_intercept(X)
        y = y.reshape(-1, 1)
            # tmp_theta is used to get the sum of thetas without the intercept part (theta[0])
        tmp_theta = np.copy(self.params[0]['theta'])
        tmp_theta[0] = 0
        gradient = [{}]
        gradient[0]['theta'] = (X_.T @ (y_pred - y) + tmp_theta @ tmp_theta) / m 
        return gradient


        

# class LogisticRegression_(LinearRegression_):
#     def __init__(self, n_input=2):
#         self.generate_params(n_input)

#     def generate_params(self, n_input):
#         self.params= [{}]
#         self.params[0]['theta'] = np.random.random((n_input + 1, 1))

#     def _sigmoid(self, x):
#         return 1 / (1 + np.exp(-x))

#     def _add_intercept(self, x):
#         return np.concatenate((np.ones((x.shape[0], 1)), x), axis= 1)

#     def forward(self, X):
#         X_ = self._add_intercept(X)
#         return self._sigmoid(X_ @ self.params[0]['theta'])

#     def backward(self, X, y, y_pred):
#         m = X.shape[1]
#         X_ = self._add_intercept(X)
#         y = y.reshape(-1, 1)
#         gradient = [{}]
#         gradient[0]['theta'] = X_.T @ (y_pred - y) / m
#         return gradient


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score , f1_score
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
import pandas as pd
data = load_iris()

x = data.data
y = data.target
x = MinMaxScaler().fit_transform(x)

X_train, X_test, y_train, y_test = train_test_split(x, y)
x_ = X_train
y_ = y_train

#print(x_.shape)

p1 = LogisticRegression_()#x_.shape[1])
#print("params before: ",p1.params)
#o1 = Optimizer_(p1)
#o1.batch_optimization(x_, y_ == 0, 100000,.1)
p1.fit(x_, y_ == 0)
#print("params after: ", p1.params)
prev1 = p1.predict(X_test)

print(prev1)

# p2 = LinearModel_(x_.shape[1])
# #print(p2.params)
# o2 = Optimizer_(p1)
# o2.batch_optimization(x_, y_ == 1, 100000,.1)
# #print(p2.params)
# prev2 = p2.forward(X_test)

# p3 = LinearModel_(x_.shape[1])
# #print(p3.params)
# o2 = Optimizer_(p1)
# o2.batch_optimization(x_, y_ == 2, 100000,.1)
# #print(p3.params)
# prev3 = p3.forward(X_test)

# prev = pd.DataFrame(np.concatenate((prev1, prev2, prev3), axis=1)).idxmax(axis=1)
# print(prev)

# met = precision_score(y_test,(prev < .5), average='macro')
# print(met)