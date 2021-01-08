from tf_compat import tf


def act_fn(x):
    return tf.nn.elu(x)


def build_mlp(mlp_input, hid_dims, output_size, prefix='', output_activation=None):
    x = mlp_input
    for i, hid_dim in enumerate(hid_dims):
        x = tf.layers.Dense(
            hid_dim,
            activation=act_fn,
            name='{}_mlp_{}'.format(prefix, i)
        )(x)
    out = tf.layers.Dense(output_size, activation=output_activation)(x)
    return out





