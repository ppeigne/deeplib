from typing import Tuple
from initialization import *
from activations import *

class Layer():
    def __init__(self, n_units, activation):#, initialization=None):
        self.n_units = n_units
        #self.activation = self._select_activation(activation)
        #self.initialization = initialization if initialization else self._select_initialization(self)

    def _select_activation(self, activation):
        raise NotImplementedError

    def _select_initilization(self):
        raise NotImplementedError


class DeepLayer(Layer):
    def __init__(self, n_units, activation='relu', dropout=0):
        self.n_units = n_units
        self.activation = self._select_activation(activation)
        self.initialization = self._select_initialization(activation)
        self.dropout_rate = dropout


    def _select_activation(self, activation):
        activations = {
            'sigmoid': sigmoid,
            # 'tanh'; ,
            'relu': relu
            # 'leaky_relu': leaky_relu,
            # 'lrelu': ,
            # 'prelu': ,
            # 'elu': ,
        }
        return activations[activation]

    def _select_initialization(self, activation):
        initializations = {
            'sigmoid': initialize_tanh,
        #    'tanh'; initialize_tanh,
            'relu': initialize_relu
            # 'leaky_relu': ,
            # 'lrelu': ,
            # 'prelu': ,
            # 'elu': ,
        }
        return initializations[activation]


class Dense(DeepLayer):
    #def _init__(self)

    def _generate_params(self, input_dim):
        parameters = {
            'W': self.initialization((self.n_units, input_dim)),
            'b': np.zeros(1),
            'Z': np.zeros(self.n_units),
            'A': np.zeros(self.n_units),
            'g': self.activation
        }
        gradients = {
            'dW': np.zeros((self.n_units, input_dim)),
            'db': np.zeros(1),
            'dZ': np.zeros(self.n_units),
            'dA': np.zeros(self.n_units),
            'dg': self.activation # FIXME 
        } 
        return parameters, gradients

class Flatten(Layer):
    def __init__(self, input_dim: Tuple[int, int]) -> None:
        self.n_units = input_dim[0] * input_dim[1]

class Input(Layer):
    def __init__(self, input_dim: int) -> None:
        self.n_units = input_dim
    
    def _generate_params(self, input_dim):
        parameters = {'A': np.zeros(self.n_units)}
        gradients = {'dA': np.zeros(self.n_units)}
        return parameters, gradients


class Network():
    def __init__(self, architecture):
        self.depth = len(architecture)
        self.params, self.gradients = self._generate_params(architecture)

    def _generate_params(self, architecture):
        params = []
        gradients = []
        for l in range(self.depth):
            input_dim = architecture[l-1].n_units
            tmp_params, tmp_gradients = architecture[l]._generate_params(input_dim)
            params.append(tmp_params)
            gradients.append(tmp_gradients)
            # params.append(architecture[l]._generate_params(input_dim))
        return params, gradients  

    def forward(self, X):
        self.params[0]['A'] = X
        for l in range(1, self.depth):
            self.params[l]['Z'] = self.params[l]['W'] @ self.params[l-1]['A'] + self.params[l]['b']
            self.params[l]['A'] = self.params[l]['g'](self.params[l]['Z'])
        return self.params[self.depth-1]['A']
    
    def gradient(self, X, y, result):
        _, m = X.shape
        self.gradients[self.depth-1]['dA'] =  self.params[self.depth-1]['A']# FIXME
        for l in range(self.depth, 0):
            self.gradients[l]['dZ'] = self.gradients[l]['dA'] * self.gradients[l]['dg'](self.params[l]['Z']) 
            self.gradients[l]['dW'] = (self.gradients[l]['dZ'] @ self.params[l-1]['A'].T) / m 
            self.gradients[l]['db'] = (np.sum(self.gradients[l]['dZ'], axis=1, keepdims=True)) / m
            self.gradients[l-1]['dA'] = self.params[l]['W'].T @ self.gradients[l]['dZ']


            
    

x = [Input(12), 
    Dense(2, activation='sigmoid', dropout=.2), 
    Dense(8), 
    Dense(2, activation='sigmoid', dropout=.5)]

model = Network(x)
#for i,p in enumerate(model.params):
    # print(f"W[{i}]",p['W'].shape)
    # print(f"A[{i}]",p['A'].shape)
    # print(i)
    # print(p)


X = np.random.random((12,10)) * 10
res = model.forward(X)
print(res)

y_ = (np.random.random((2, 10)) > .5) * 1

g = model.gradient(X, y_, res)