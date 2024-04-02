import random
import time
from collections import namedtuple, deque

import gymnasium as gym
import numpy as np
import tensorflow as tf
import utils

SEED = 0
random.seed(SEED)
NUM_STEPS_FOR_UPDATE = 4
GAMMA = 0.995  # discount factor
ALPHA = 1e-3  # learning rate


# 训练强化学习策略，完成lunar lander任务

def init_network(num_states=8, num_actions=4):
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=num_states),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(num_actions, activation="linear")
    ])


# 初始化Q网络和目标Q网络
q_network = init_network()
target_q_network = init_network()
optimizer = tf.keras.optimizers.Adam(learning_rate=ALPHA)

def compute_loss(experiences, gamma, q_network, target_q_network):
    """
    Calculate the loss of the network

    :param experiences:
    :param gamma:
    :param q_network:
    :param target_q_network:
    :return:
    """
    # Unpack the mini-batch of experience tuples
    states, actions, rewards, next_states, done_vals = experiences
    # Compute max Q^(s,a)
    # find the max q value for each sample in the batch
    max_qsa = tf.reduce_max(target_q_network(next_states), axis=-1)

    # Set y = R if episode terminates, otherwise set y = R + γ max Q^(s,a).
    y_targets = rewards + (gamma * max_qsa * (1 - done_vals))

    q_values = q_network(states)
    q_values = tf.gather_nd(q_values, tf.stack([tf.range(q_values.shape[0]),
                                                tf.cast(actions, tf.int32)], axis=1))
    # Compute the loss
    loss = tf.keras.losses.MSE(y_targets, q_values)

    return loss


# @tf.function
def agent_learn(experiences, gamma):
    """
    Updates the weights of the Q networks.

    Args:
      experiences: (tuple) tuple of ["state", "action", "reward", "next_state", "done"] namedtuples
      gamma: (float) The discount factor.
      q_network: (tf.keras.Model) The Q-Network.
      target_q_network: (tf.keras.Model) The target Q-Network.
      optimizer: (tf.keras.optimizers.Optimizer) The optimizer to use for updating the Q-Network.
    """

    # Calculate the loss
    with tf.GradientTape() as tape:
        loss = compute_loss(experiences, gamma, q_network, target_q_network)

    # Get the gradients of the loss with respect to the weights.
    gradients = tape.gradient(loss, q_network.trainable_variables)

    # Update the weights of the q_network.
    optimizer.apply_gradients(zip(gradients, q_network.trainable_variables))

    # update the weights of target q_network
    utils.update_target_network(q_network, target_q_network)


def get_action(q_values, epsilon=0.0):
    """
    Returns an action using an ε-greedy policy.

    This function will return an action according to the following rules:
        - With probability epsilon, it will return an action chosen at random.
        - With probability (1 - epsilon), it will return the action that yields the
        maximum Q value in q_values.

    Args:
        q_values (tf.Tensor):
            The Q values returned by the Q-Network. For the Lunar Lander environment
            this TensorFlow Tensor should have a shape of [1, 4] and its elements should
            have dtype=tf.float32.
        epsilon (float):
            The current value of epsilon.

    Returns:
       An action (numpy.int64). For the Lunar Lander environment, actions are
       represented by integers in the closed interval [0,3].
    """
    # With probability epsilon, choose an action at random.
    if np.random.random() < epsilon:
        act = np.random.randint(0, 4)
    else:
        # With probability (1 - epsilon), choose the action that
        # yields the maximum Q value.
        act = np.argmax(q_values.numpy()[0])

    return act


if __name__ == "__main__":
    env = gym.make("LunarLander-v2", render_mode="human")
    observation, info = env.reset()
    num_states = env.observation_space.shape[0]
    num_actions = env.action_space.n

    experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    target_q_network.set_weights(q_network.get_weights())
    epsilon = 1.0
    # Create a memory buffer D with capacity N
    memory_buffer = deque(maxlen=100_000)
    # start Deep Q-learning

    # train it 2000 episodes
    num_episodes = 2000
    # the max number of steps in each episode
    max_time_steps = 1000
    total_point_history = []

    num_p_av = 100
    start = time.time()
    for i in range(num_episodes):

        # Reset the environment to the initial state and get the initial state
        observation = env.reset()
        state = observation[0]
        total_points = 0

        for t in range(max_time_steps):

            # From the current state S choose an action A using an ε-greedy policy
            state_qn = np.expand_dims(state, axis=0)  # state needs to be the right shape for the q_network
            q_values = q_network(state_qn)
            action = get_action(q_values, epsilon)

            # Take action A and receive reward R and the next state S'
            next_state, reward, terminated, truncated, _ = env.step(action)
            if truncated:
                print("Truncated")

            done = terminated

            # Store experience tuple (S,A,R,S') in the memory buffer.
            # We store the done variable as well for convenience.
            memory_buffer.append(experience(state, action, reward, next_state, done))

            # Only update the network every NUM_STEPS_FOR_UPDATE time steps.
            update = utils.check_update_conditions(t, NUM_STEPS_FOR_UPDATE, memory_buffer)

            if update:
                # Sample random mini-batch of experience tuples (S,A,R,S') from D
                experiences = utils.get_experiences(memory_buffer)

                # Set the y targets, perform a gradient descent step,
                # and update the network weights.
                agent_learn(experiences, GAMMA)

            state = next_state.copy()
            total_points += reward

            if done:
                break

        total_point_history.append(total_points)
        av_latest_points = np.mean(total_point_history[-num_p_av:])

        # Update the ε value
        epsilon = utils.get_new_eps(epsilon)

        print(f"\rEpisode {i + 1} | Total point average of the last {num_p_av} episodes: {av_latest_points:.2f}",
              end="")

        if (i + 1) % num_p_av == 0:
            print(f"\rEpisode {i + 1} | Total point average of the last {num_p_av} episodes: {av_latest_points:.2f}")

        # We will consider that the environment is solved if we get an
        # average of 200 points in the last 100 episodes.
        if av_latest_points >= 200.0:
            print(f"\n\nEnvironment solved in {i + 1} episodes!")
            q_network.save('lunar_lander_model.h5')
            break

    tot_time = time.time() - start

    print(f"\nTotal Runtime: {tot_time:.2f} s ({(tot_time / 60):.2f} min)")
