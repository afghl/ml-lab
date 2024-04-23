import math
import numpy as np
import activation_function as f

activation_functions = {
    'relu': f.ReLU,
    'id': f.Identity,
}


class Layer(object):

    def forward_pass(self, X, training):
        raise NotImplementedError()

    def backward_pass(self, grad):
        raise NotImplementedError()


class Dense(Layer):

    def __init__(self, n_units, input_shape: tuple, activation, learning_rate=0.01):
        self.activation_func = activation_functions[activation]
        self.n_units = n_units
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.layer_input = None
        self.initialize_weights_and_bias()

    def initialize_weights_and_bias(self):
        limit = 1 / math.sqrt(self.input_shape[0])
        # weights size should be (n^[l], n^[l-1])
        self.W = np.random.uniform(limit, -limit, (self.input_shape[0], self.n_units))
        self.b = np.zeros((self.n_units, 1))

    def forward_pass(self, X, training):
        self.layer_input = X
        z = np.dot(X, self.W) + self.b
        return self.activation_func(z)

    def backward_pass(self, a_grad):
        z_grad = a_grad * self.activation_func.gradient(self.layer_input)
        grad_W = np.dot(self.layer_input.T, z_grad)
        grad_b = np.sum(z_grad, axis=0, keepdims=True)
        grad = np.dot(z_grad, self.W.T)
        self.W = self.W - self.learning_rate * grad_W
        self.b = self.b - self.learning_rate * grad_b
        return grad
