import os

import numpy as np

from deep_learning.neuron_network import NeuronNetwork
from deep_learning.layer import Dense
from deep_learning.loss_functions import LossFunction, MeanSquaredError

def load_mnist(data_path='../data/mnist.npz'):
    # print pwd
    print(os.listdir('.'))
    # 从本地文件加载数据
    with np.load(data_path) as data:
        train_images, train_labels = data['x_train'], data['y_train']
        test_images, test_labels = data['x_test'], data['y_test']
    return (train_images, train_labels), (test_images, test_labels)


def main():
    (x_train, y_train), (x_test, y_test) = load_mnist()
    print(f"训练集图像形状：{x_train.shape}")
    print(f"训练集标签形状：{y_train.shape}")
    print(f"测试集图像形状：{x_test.shape}")
    print(f"测试集标签形状：{y_test.shape}")

    model = NeuronNetwork(loss=MeanSquaredError)

    model.add(Dense(512, activation='relu', input_shape=(784,)))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.summary()


if __name__ == '__main__':
    main()
