
# UNQ_C1
# GRADED CELL: my_softmax
import numpy as np


def my_softmax(z):
    """ Softmax converts a vector of values to a probability distribution.
    Args:
      z (ndarray (N,))  : input data, N features
    Returns:
      a (ndarray (N,))  : softmax of z
    """
    ### START CODE HERE ###
    a = np.zeros(z.shape)

    sum_of_z = 0
    for i in range(z.shape[0]):
        sum_of_z += np.exp(z[i])
    for i in range(z.shape[0]):
        a[i] = np.exp(z[i]) / sum_of_z

    ### END CODE HERE ###
    return a

def get_tenser_flow_model():
    # UNQ_C2
    # GRADED CELL: Sequential model
    tf.random.set_seed(1234)  # for consistent results
    model = Sequential(
        [
            ### START CODE HERE ###
            Dense(25, activation='relu', input_shape=(400,)),
            Dense(15, activation='relu', input_shape=(25,)),
            Dense(10, activation='linear', input_shape=(15,)),
            ### END CODE HERE ###
        ], name="my_model"
    )
    return model


def compute_loss(experiences):
    states, actions, rewards, next_states, done_vals = experiences

    # experiences: (tuple) tuple of ["state", "action", "reward", "next_state", "done"] namedtuples
    # done_vals: (tensor) tensor of bools of whether the episode is complete or not
    # first, compute y: if episode is done, y = reward, else y = reward + gamma * max_a' Q(s', a')
    y = rewards + (1 - done_vals) * gamma * np.max(model.predict(next_states), axis=1)

    loss = tf.keras.losses.MSE(y, model.predict(states))