import numpy as np


class NeuronNetwork(object):
    def __init__(self, loss):
        self.layers = []
        self.loss_func = loss

    def add(self, layer):
        # 除了第一层的layer，之后的layer都根据前一层的output来计算input_shape
        if len(self.layers) > 0:
            last = self.layers[-1]
            layer.set_input_shape(last.output_shape())
        if hasattr(layer, 'initialize'):
            layer.initialize()
        self.layers.append(layer)

    def predict(self, X):
        return self._forward(X, training=False)

    def fix(self, X, y, n_epochs):
        for i in range(n_epochs):
            y_pred = self._forward(X, training=True)
            loss = np.mean(self.loss_func.loss(y, y_pred))
            grad = self.loss_func.gradient(y, y_pred)
            self._backward(grad)

    def _forward(self, X, training):
        out = X
        for layer in self.layers:
            out = layer.forward_pass(out, training)
        return out

    def _backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward_pass(grad)

    def summary(self):
        # print all layers
        for i, layer in enumerate(self.layers):
            print("Layer {}: {}, Input shape: {}, Output shape: {}, "
                  "W.shape: {}, b.shape: {}".format(i, layer.__class__.__name__,
                                                    layer.input_shape, layer.output_shape(), layer.W.shape,
                                                    layer.b.shape))
