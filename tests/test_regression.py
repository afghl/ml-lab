import math
import numpy as np
import pytest
from supervised_learning.regression import Regression


# 测试 Regression 类的 initialize_weights_and_bias 方法
def test_initialize_weights_and_bias():
    r = Regression(n_iterations=100)
    n_features = 5
    r.initialize_weights_and_bias(n_features)
    # 检查权重矩阵的形状是否正确
    assert r.w.shape == (n_features,)
    # 检查权重矩阵的值是否在合理范围内
    assert np.all(r.w >= -1 / math.sqrt(n_features))
    assert np.all(r.w <= 1 / math.sqrt(n_features))
