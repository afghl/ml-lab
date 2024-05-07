import numpy as np


class NeuronNetwork(object):
    def __init__(self, loss):
        self.layers = []
        self.loss_func = loss()
        self.errors = []

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

    def fit(self, X, y, n_epochs, batch_size):
        for i in range(n_epochs):
            batch_error = []
            for X_batch, y_batch in batch_iterator(X, y, batch_size):
                y_pred = self._forward(X_batch, training=True)
                loss = np.sum(self.loss_func.loss(y_batch, y_pred))
                batch_error.append(loss)
                grad = self.loss_func.gradient(y_batch, y_pred)
                self._backward(grad)
            self.errors.append(np.mean(batch_error))
            print("Epoch: {}, Loss: {:.3f}".format(i, np.mean(batch_error)))

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


def batch_iterator(X, y=None, batch_size=64):
    """ Simple batch generator """
    n_samples = X.shape[0]
    for i in np.arange(0, n_samples, batch_size):
        begin, end = i, min(i+batch_size, n_samples)
        if y is not None:
            yield X[begin:end], y[begin:end]
        else:
            yield X[begin:end]