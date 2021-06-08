import numpy as np
from typing import Tuple
from initialization import *
from activations import *

class Layer():
    def __init__(self, n_units):#, initialization=None):
        self.n_units = n_units
        #self.activation = self._select_activation(activation)
        #self.initialization = initialization if initialization else self._select_initialization(self)

    def _select_activation(self, activation):
        raise NotImplementedError

    def _select_initilization(self, activation):
        raise NotImplementedError

# class Activation(Layer):
#     def __init__(self, activation='relu'):
#         self.activation = self._select_activation(activation)

#     def _select_activation(self, activation):
#         activations = {
#             'relu': relu
#         }
#         return activations[activation]

class DeepLayer(Layer):
    def __init__(self, n_units, activation='relu'):
        self.n_units = n_units
        self.activation = self._select_activation(activation)
        self.activation_prime = self._select_activation_prime(activation)
        self.initialization = self._select_initialization(activation)

    def _select_activation(self, activation):
        activations = {
            # 'sigmoid': sigmoid,
            # 'tanh'; ,
            'relu': relu
            # 'leaky_relu': leaky_relu,
            # 'lrelu': ,
            # 'prelu': ,
            # 'elu': ,
        }
        return activations[activation]

    def _select_activation_prime(self, activation):
        activations = {
            # 'sigmoid': sigmoid_prime,
            # 'tanh'; ,
            'relu': relu_prime
            # 'leaky_relu': leaky_relu,
            # 'lrelu': ,
            # 'prelu': ,
            # 'elu': ,
        }
        return activations[activation]

    def _select_initialization(self, activation):
        initializations = {
            # 'sigmoid': initialize_tanh,
            # 'tanh'; initialize_tanh,
            'relu': initialize_relu
            # 'leaky_relu': ,
            # 'lrelu': ,
            # 'prelu': ,
            # 'elu': ,
        }
        return initializations[activation]


class Dense(DeepLayer):
    def _init__(self, n_units, activation):
        super().__init__(self, n_units, activation)
        self.params = None
        self.grads = None

    def _generate_params(self, input_dim):
        self.params = {
            'W': self.initialization((self.n_units, input_dim)),
            'b': np.zeros(1),
            'Z': np.zeros(self.n_units),
            'A': np.zeros(self.n_units),
            'g': self.activation
        }
        self.grads = {
            'dW': np.zeros((self.n_units, input_dim)),
            'db': np.zeros(1),
            'dZ': np.zeros(self.n_units),
            'dA': np.zeros(self.n_units),
            'dg': self.activation # FIXME 
        } 
        return self.params, self.grads

    def forward(self, A_prev):
        self.A_prev = A_prev
        self.params['Z'] = self.params['W'] @ A_prev + self.params['b']
        self.params['A'] = self.params['g'](self.params['Z'])
        return self.params['A'] # IS IT NECESSARY ?

    def backward(self, dA, m):
        self.grads['dZ'] = dA * self.params['dg'](self.params['Z']) 
        self.grads['dW'] = self.grads['dZ'] @ self.A_prev.T / m 
        self.grads['db'] = np.sum(self.grads['dZ'], axis=1, keepdims=True) / m
        self.grads['dA'] = self.params['W'].T @ self.grads['dZ']
        return self.grads['dA'] # IS IT NECESSARY ?
 
    # def update(self, update_rule):
    #     for k in self.params.keys():
    #         self.params[k] = update_rule(self.params[k], self.grads[f"d{k}"])


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

class Output(Layer):
    def __init__(self, input_dim: int) -> None:
        self.n_units = input_dim

    def _generate_params(self, input_dim):
        parameters = {'A': np.zeros(self.n_units)}
        gradients = {'dA': np.zeros(self.n_units)}
        return parameters, gradients
    
    def forward(self, x):
        return 1

    def backward(self):
        return 1

class Network():
    def __init__(self, architecture):
        self.depth = len(architecture)
        self.architecture = architecture
        self.params, self.gradients = self._generate_params()

    # def _generate_params(self, architecture):
    #     for l in range(self.depth):
    #         input_dim = architecture[l-1].n_units
    #         architecture[l]._generate_params(input_dim)

    def _generate_params(self):
        params = []
        gradients = []
        for l in range(self.depth):
            input_dim = self.architecture[l-1].n_units
            tmp_params, tmp_gradients = self.architecture[l]._generate_params(input_dim)
            params.append(tmp_params)
            gradients.append(tmp_gradients)
        return params, gradients  

    def forward(self, X):
        layer_input = X
        for l in range(1, self.depth):
            layer_input = self.architecture[l].forward(layer_input)
        return layer_input
    
    def gradient(self, X, y, result):
        _, m = X.shape
        dA = self.architecture[self.depth - 1].backward()#gradient # FIXME
        for l in range(self.depth - 1, 0):
            dA = self.architecture[l].backward(dA, m)
        return dA


            
    

x = [Input(12), 
    Dense(2, activation='relu'), 
    Dense(8), 
    Dense(2, activation='relu'),
    Output(3)]

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
print(g)