import numpy as np


def my_dense(a_in, W, b, g):
    """
    Computes dense layer
    Args:
      a_in (ndarray (n, )) : Data, 1 example
      W    (ndarray (n,j)) : Weight matrix, n features per unit, j units
      b    (ndarray (j, )) : bias vector, j units
    Returns
      a_out (ndarray (j,))  : j units|
    """
    units = W.shape[1]
    a_out = np.zeros(units)
    for j in range(units):
        w = W[:,j]
        z = np.dot(w, a_in) + b[j]
        a_out[j] = g(z)
    print("after compute, W is ", W, "\na_in is ", a_in,  "\na_out is", a_out)
    return a_out


def new_my_dense(a_in, W, b, g):
    """
    Computes dense layer
    Args:
      a_in (ndarray (n, )) : Data, 1 example
      W    (ndarray (n,j)) : Weight matrix, n features per unit, j units
      b    (ndarray (j, )) : bias vector, j units
    Returns
      a_out (ndarray (j,))  : j units|
    """
    # units = W.shape[1]
    # a_out = np.zeros(units)
    print("before compute, W is ", W, "\na_in is ", a_in)
    z = a_in @ W
    z = z + b
    print("after compute, W is ", W, "\na_in is ", a_in,  "\na_out is", z)
    return z


