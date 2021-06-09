import numpy as np
#from layers import Network
from typing import Tuple, List
import copy


class Optimizer():
    def __init__(self, batch_size: int = -1):
        self.epoch = 0
        self.batch_size = batch_size
        self.optimize = self.batch_optimization if batch_size == -1 else self.minibatch_optimization

    def _generate_params(self, model_params):
        pass

    def batch_optimization(self, model, X: np.ndarray, y: np.ndarray, 
                            n_cycles: int = 10000, learning_rate: float = 0.1, 
                            learning_rate_decay: bool = False):
        self._generate_params(model.params)
        for self.epoch in range(n_cycles):
            results = model.predict(X)
            model.gradient(X, y, results)
            model.params = self.update(model.params, model.gradients, learning_rate, learning_rate_decay)
        return model


    def minibatch_optimization(self, model, X: np.ndarray, y: np.ndarray, 
                                n_cycles: int = 10000, learning_rate: float = 0.1, 
                                learning_rate_decay: bool = False):
        self._generate_params(model.params)
        m, _ = X.shape
        for self.epoch in range(n_cycles):
            for t in range(m // self.batch_size):
                mini_batch = X[:, self.batch_size * t:self.batch_size * (t + 1)]     
                results = model.forward(mini_batch, model.params)
                gradient = model.backward(mini_batch, results)
                model.params = self.update(model.params, gradient, learning_rate, learning_rate_decay)
            rest = m % self.batch_size
            if rest != 0:
                final_batch = X[:, -rest:]
                results = model.forward(final_batch, model.params)
                gradient = model.backward(final_batch, results)
                model.params = self.update(model.params, gradient, learning_rate, learning_rate_decay)
        return model

    def update(self, params: List[dict], gradient: List[dict], 
                learning_rate: float, learning_rate_decay: bool):
        for l in range(1, len(params)):
            for param in params[l].keys():
                if learning_rate_decay: 
                    param_update = self._update_rule(gradient, l, param, learning_rate ** self.epoch)
                else:
                    param_update = self._update_rule(gradient, l, param, learning_rate)
                params[l][param] += param_update
        return params

    def _update_rule(self, gradient,l, param, learning_rate):
        return - gradient[l][f'd{param}'] * learning_rate


class MomentumOptimizer(Optimizer):
    def __init__(self, batch_size=-1, beta=0.9, gamma=None, bias_correction=False):
        super().__init__(batch_size)
        self.beta = beta
        self.gamma = gamma if gamma else 1 - beta
        self.bias_correction = bias_correction
        self.update_params = []

    def _generate_params(self, model_params: List[dict]):
        for l in range(len(model_params)):
            self.update_params.append({})
            for param in model_params[l].keys():
                self.update_params[l][param] = np.zeros_like(model_params[l][param])

    def _update_rule(self, gradient,l, param, learning_rate):
        self.update_params[l][param] *= self.beta 
        self.update_params[l][param] += self.gamma * gradient[l][f'd{param}']
        if self.bias_correction: 
            self.update_params[l][param] / (1 - (self.beta ** self.epoch) + 1e-8)
        return - self.update_params[l][param] * learning_rate


class RMSOptimizer(Optimizer):
    def __init__(self, batch_size=-1, beta=0.99, bias_correction=False):
        super().__init__(batch_size)
        self.beta = beta
        self.bias_correction = bias_correction
        self.update_params = []

    def _generate_params(self, model_params: List[dict]):
        for l in range(len(model_params)):
            self.update_params.append({})
            for param in model_params[l].keys():
                self.update_params[l][param] = np.zeros_like(model_params[l][param])

    def _update_rule(self, gradient,l, param, learning_rate):
        self.update_params[l][param] *= self.beta 
        self.update_params[l][param] += (1 - self.beta) * (gradient[l][f'd{param}']**2)
        if self.bias_correction: 
            self.update_params[l][param] / (1 - (self.beta ** self.epoch) + 1e-8)
        return - gradient[l][f'd{param}'] / (self.update_params[l][param] + 1e-8) * learning_rate


class AdamOptimizer(Optimizer):
    def __init__(self, batch_size=-1, beta1=0.9, beta2=0.99):
        super().__init__(batch_size)
        self.beta = (beta1, beta2)
        self.update_params = []
        

    def _generate_params(self, model_params: List[dict]):
        for l in range(len(model_params)):
            self.update_params.append({})
            for param in model_params[l].keys():
                self.update_params[l][param] = [np.zeros_like(model_params[l][param]), 
                                                np.zeros_like(model_params[l][param])]
    
    def _update_rule(self, gradient,l, param, learning_rate):
        # Momentum part
        self.update_params[l][param][0] *= self.beta[0] 
        self.update_params[l][param][0] += (1 - self.beta[0]) * gradient[l][f'd{param}']
        momentum_corrected = self.update_params[l][param][0] / (1 - (self.beta[0] ** self.epoch) + 1e-8)
        # RMS part
        self.update_params[l][param][1] *= self.beta[1] 
        self.update_params[l][param][1] += (1 - self.beta[1]) * (gradient[l][f'd{param}']**2)
        rms_corrected = self.update_params[l][param][1] / (1 - (self.beta[1] ** self.epoch) + 1e-8)
        return - momentum_corrected / (np.sqrt(rms_corrected + 10e-8)) * learning_rate
        

