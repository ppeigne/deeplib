import numpy as np
from typing import Tuple, List
from initialization import  * #initialize_relu
from activations import * #relu, relu_prime
from optimizers import Optimizer_


class Layer():
    def __init__(self, n_units: int):#, initialization=None):
        self.n_units = n_units

    def _select_activation(self, activation):
        raise NotImplementedError

    def _select_initilization(self, activation):
        raise NotImplementedError


class DeepLayer(Layer):
    def __init__(self, n_units: int, activation: str = 'relu'):
        self.n_units = n_units
        self.activation = self._select_activation(activation)
        self.activation_prime = self._select_activation_prime(activation)
        self.initialization = self._select_initialization(activation)

    def _select_activation(self, activation: str):
        activations = {
            'sigmoid': sigmoid,
            'relu': relu,
            # 'softmax': softmax
            # 'leaky_relu': leaky_relu,
        }
        return activations[activation]

    def _select_activation_prime(self, activation: str):
        activations = {
            'sigmoid': sigmoid_prime,
            'relu': relu_prime,
            # 'softmax': softmax_prime
            # 'leaky_relu': leaky_relu,
        }
        return activations[activation]

    def _select_initialization(self, activation: str):
        initializations = {
            'sigmoid': initialize_sigmoid,
            'relu': initialize_relu,
            'softmax': initialize_relu # FIXME
            # 'leaky_relu': ,
        }
        return initializations[activation]


class Dense(DeepLayer):
    def _init__(self, n_units: int, activation: str = 'relu'):
        super().__init__(self, n_units, activation)
        self.params = None
        self.grads = None

    def _generate_params(self, input_dim: int):
        self.params = {
            'W': self.initialization((self.n_units, input_dim)),
            'b': np.zeros(1)
        }
        self.grads = {
            'dW': np.zeros((self.n_units, input_dim)),
            'db': np.zeros(1)
        } 
        self.cache = {
            'Z': None, #np.zeros(self.n_units),
            'A': None, #np.zeros(self.n_units),
            'g': self.activation,
            'dZ': None, #np.zeros(self.n_units),
            'dA': None, #np.zeros(self.n_units),
            'dg': self.activation_prime  
        }
        return self.params, self.grads

    def forward(self, A_prev: np.ndarray):
        # A_prev = A_prev
        self.cache['Z'] = self.params['W'] @ A_prev + self.params['b']
        self.cache['A'] = self.cache['g'](self.cache['Z'])
        return self.cache['A'] 

    def backward(self, dA: np.ndarray, m: int):
        self.cache['dZ'] = dA * self.cache['dg'](self.cache['Z']) 
        self.grads['dW'] = self.cache['dZ'] @ self.A_prev.T / m 
        self.grads['db'] = np.sum(self.cache['dZ'], axis=1, keepdims=True) / m
        self.cache['dA'] = self.params['W'].T @ self.cache['dZ']
        return self.cache['dA']


class Dropout(Layer):
    def __init__(self, keep_prob: int = .5):
        self.keep_prob = keep_prob
        self.training = True
        self.params = {}
        self.grads = {}
    
    def _generate_params(self, input_dim: int):
        self.cache = {'drop_matrix': None}
        self.n_units = input_dim
        return self.params, self.grads

    def forward(self, A_prev: np.ndarray):
        if self.training:
            self.cache['drop_matrix'] = np.random.uniform(0, 1,  A_prev.shape) < self.keep_prob
            return A_prev * self.cache['drop_matrix'] / (1. - self.keep_prob)
        return A_prev
    
    def backward(self, dA: np.ndarray, m: int):
        return dA * self.drop_matrix / (1. - self.keep_prob)

class BatchNormalization(Layer):
    def __init__(self):
        pass

    def _generate_params(self, input_dim: int):
        self.params = {
            'gamma': 0,
            'beta': 0
        }
        self.grads = {
            'dgamma': 0,
            'dbeta': 0 
        } 
        return self.params, self.grads

    def forward(self, A_prev: np.ndarray, eps: float = 1e-10):
        mu = A_prev.mean(axis=0)
        variance = A_prev.variance(axis=0)
        X_norm = (A_prev - mu) / (variance + eps) 
        return (self.params['gamma'] * X_norm) + self.params['beta']

    def backward(self, dA: np.ndarray):
        self.params['gamma'] = dA
        self.params['beta'] = np.sum(dA, axis=1, keepdims=True)


class Flatten(Layer):
    def __init__(self, n_units: Tuple[int, int]) -> None:
        self.n_units = n_units[0] * n_units[1]

# class Input(Layer):
#     def __init__(self, n_units: int) -> None:
#         self.n_units = n_units
#         # self.params, self.gradients = self._generate_params()
    
#     def _generate_params(self, input_dim):
#         parameters = {'A': np.zeros(self.n_units)}
#         gradients = {'dA': np.zeros(self.n_units)}
#         return parameters, gradients

# class Output(Layer):
#     def __init__(self, n_units: int) -> None:
#         self.n_units = n_units

#     def _generate_params(self, input_dim):
#         parameters = {'A': np.zeros(self.n_units)}
#         gradients = {'dA': np.zeros(self.n_units)}
#         return parameters, gradients
    
#     def forward(self, x):
#         return 1

#     def backward(self):
#         return 1

class Network():
    def __init__(self, architecture: List[Layer], loss: str ='cross_entropy'):
        self.depth = len(architecture)
        self.architecture = architecture
        self.params, self.gradients = self._generate_params()
        self.loss, self.loss_prime = self._select_loss(loss)

    def _select_loss(self, loss: str):
        losses = {
            'cross_entropy' : (None, lambda y_true, y_pred: y_true - y_pred)
        }
        return losses[loss]  

    def _generate_params(self):
        params = []
        gradients = []
        for l in range(1, self.depth):
            input_dim = self.architecture[l-1].n_units
            tmp_params, tmp_gradients = self.architecture[l]._generate_params(input_dim)
            params.append(tmp_params)
            gradients.append(tmp_gradients)
        return params, gradients  

    def predict(self, X: np.ndarray):
        return self.forward(X)

    def forward(self, X: np.ndarray):
        layer_input = X
        for l in range(1, self.depth):
            layer_input = self.architecture[l].forward(layer_input)
        return layer_input
    
    def gradient(self, X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray):
        _, m = X.shape
        dA = self.loss_prime(y_true, y_pred) #self.architecture[self.depth - 1].backward()#gradient # FIXME
        for l in range(self.depth - 1, -1):
            dA = self.architecture[l].backward(dA, m)
        return dA

    def train(self, X: np.ndarray, y: np.ndarray, verbose=False):
        optimizer = Optimizer_()
        optimizer.optimize(self, X, y)



# x = Dense(3)  
# print(x.cache)

architecture = [Dense(3, activation='relu'), 
                Dropout(keep_prob=.8),
                Dense(8), 
                Dropout(keep_prob=.8),
                Dense(1, activation='sigmoid')] 

model = Network(architecture)

# # for l in model.architecture[1:]:
# #     print(l)
# #     print(l.params)
# #     print(l.grads, end='\n\n')

X = np.array([[1, 0, 1],
              [3, 2, 2],
              [0, 9, 1]])
# res = model.forward(X)
# print(f"prediction = {res}")
# print(f"model params = {model.params}")

y_ = np.array([[1], [0], [1]])

# g = model.gradient(X, y_, res)
# print(g)
# print(f"model grads = {model.gradients}")

model.train(X, y_)

print(model.predict(X))

print(model.params)