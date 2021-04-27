import numpy as np
from optimizers import Optimizer_

class Perceptron():
    def __init__(self, n_input=2):
        self.generate_params(n_input)

    def generate_params(self, n_input):
        self.params= [{}]
        self.params[0]['W'] = np.random.random((1, n_input))
        self.params[0]['b'] = np.random.random(1)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def predict(self, X):
        return self._sigmoid((self.params[0]['W'] @ X) + self.params[0]['b'])

    def gradient(self, X, y, y_pred):
        m = X.shape[1]
        X_ = np.concatenate((np.ones((1, m)), X), axis=0)
        grad = X_ @ (y_pred - y).T / m
        gradient = [{}]
        gradient[0]['b'] = grad[0]
        gradient[0]['W'] = grad[1:].T 
        return gradient

#class NeuralNetwork_():

architecture = [(5, 'relu'), (2, 'relu')]
def generate_params(n_inputs, architecture):
    architecture = [(n_inputs, None)] + architecture
    params = [{}]
    for l in range(1, len(architecture)):
        ### FIXME
        params[l-1]['W'] = np.random.random((architecture[l - 1][0], architecture[l][0]))
        params[l-1]['b'] = np.random.random(1)
    return params

x = generate_params(10, architecture)
print(x)


