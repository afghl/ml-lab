import math
import numpy as np
import pytest
from deep_learning.neuron_network import NeuronNetwork


# 测试 Regression 类的
def test_initialize_nn():
    model = NeuronNetwork()
    # assert not null
    assert model.layers == []