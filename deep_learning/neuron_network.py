import numpy as np


class NeuronNetwork(object):

    def __init__(self, loss):
        self.layers = []
        self.loss_func = loss

    def add(self, layer):
        self.layers.append(layer)

    def predict(self, X):
        self._forward(X, training=False)

    def fix(self, X, y, n_epochs):
        for i in range(n_epochs):
            y_pred = self._forward(X, training=True)
            loss = np.mean(self.loss_func.loss(y, y_pred))
            grad = self.loss_func.gradient(y, y_pred)
            self._backward(grad)

    def _forward(self, X, training):
        out = X
        for layer in self.layers:
            out = layer.forward_pass(out)
        return out

    def _backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward_pass(grad)
