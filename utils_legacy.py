# utils_legacy.py
import tensorflow as tf


def legacy_dense(x, units, name, activation=None):
    """
    Manually implement a dense layer using tf.compat.v1.get_variable,
    so that variable names match the old TF1-style checkpoint, e.g.
    main/pi/dense/kernel and main/pi/dense/bias.
    """
    with tf.compat.v1.variable_scope(name):
        in_dim = x.get_shape().as_list()[-1]
        if in_dim is None:
            raise ValueError("Input dimension to legacy_dense is None.")

        w = tf.compat.v1.get_variable(
            "kernel",
            shape=[in_dim, units],
            initializer=tf.compat.v1.initializers.glorot_uniform(),
        )
        b = tf.compat.v1.get_variable(
            "bias",
            shape=[units],
            initializer=tf.zeros_initializer(),
        )

        y = tf.matmul(x, w) + b
        if activation is not None:
            y = activation(y)
        return y


def mlp_actor_critic(x,
                     a,
                     hidden_sizes=(64, 64, 64, 64),
                     activation=tf.nn.relu,
                     output_activation=tf.nn.tanh,
                     action_space=None):
    """
    Legacy version of actor-critic network for loading the old TF1 checkpoint.

    Variable naming is chosen to match the checkpoint:
      - main/pi/dense, dense_1, dense_2, dense_3, dense_4
      - main/q/dense,  dense_1, dense_2, dense_3, dense_4
    """

    assert action_space is not None, "action_space must be provided."

    act_dim = action_space.shape[0]
    act_limit = action_space.high[0]

    # ==================== policy network pi(s) ====================
    with tf.compat.v1.variable_scope('pi'):
        net = legacy_dense(
            x,
            hidden_sizes[0],
            name='dense',
            activation=activation,
        )
        net = legacy_dense(
            net,
            hidden_sizes[1],
            name='dense_1',
            activation=activation,
        )
        net = legacy_dense(
            net,
            hidden_sizes[2],
            name='dense_2',
            activation=activation,
        )
        net = legacy_dense(
            net,
            hidden_sizes[3],
            name='dense_3',
            activation=activation,
        )
        # Output layer: 1D action, apply tanh, then multiply by act_limit
        mu = legacy_dense(
            net,
            act_dim,
            name='dense_4',
            activation=output_activation,
        )
        pi = act_limit * mu

    # ==================== Q(s,a) network ====================
    with tf.compat.v1.variable_scope('q'):
        # From the checkpoint, the first Q layer has kernel shape [8, 64]
        # This means the input is [obs(7), act(1)] concatenated
        q_input = tf.concat([x, a], axis=-1)

        q_net = legacy_dense(
            q_input,
            hidden_sizes[0],
            name='dense',
            activation=activation,
        )
        q_net = legacy_dense(
            q_net,
            hidden_sizes[1],
            name='dense_1',
            activation=activation,
        )
        q_net = legacy_dense(
            q_net,
            hidden_sizes[2],
            name='dense_2',
            activation=activation,
        )
        q_net = legacy_dense(
            q_net,
            hidden_sizes[3],
            name='dense_3',
            activation=activation,
        )
        q = legacy_dense(
            q_net,
            1,
            name='dense_4',
            activation=None,
        )
        # Squeeze to shape [batch]
        q = tf.squeeze(q, axis=1)

    # ==================== Q(s, pi(s)) network ====================
    # Reuse the same scope 'q' with reuse=True
    with tf.compat.v1.variable_scope('q', reuse=True):
        q_pi_input = tf.concat([x, pi], axis=-1)

        q_pi_net = legacy_dense(
            q_pi_input,
            hidden_sizes[0],
            name='dense',
            activation=activation,
        )
        q_pi_net = legacy_dense(
            q_pi_net,
            hidden_sizes[1],
            name='dense_1',
            activation=activation,
        )
        q_pi_net = legacy_dense(
            q_pi_net,
            hidden_sizes[2],
            name='dense_2',
            activation=activation,
        )
        q_pi_net = legacy_dense(
            q_pi_net,
            hidden_sizes[3],
            name='dense_3',
            activation=activation,
        )
        q_pi = legacy_dense(
            q_pi_net,
            1,
            name='dense_4',
            activation=None,
        )
        q_pi = tf.squeeze(q_pi, axis=1)

    return pi, q, q_pi



