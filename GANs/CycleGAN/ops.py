import tensorflow as tf

## ops alias
relu = tf.nn.relu
lrelu = tf.nn.leaky_relu
sigmoid = tf.nn.sigmoid

## layers alias
bn = tf.layers.batch_normalization
conv2d = tf.layers.conv2d
conv2d_t = tf.layers.conv2d_transpose
dense = tf.layers.dense
flatten = tf.layers.flatten

# global varible
is_training = tf.placeholder(tf.bool, name='is_training')
global_step = tf.train.get_or_create_global_step()


# cyclegan的训练的batch_size为1，所以是不可以用batch_norm
def instance_norm(input, name="instance_norm"):
    with tf.variable_scope(name):
        depth = input.get_shape()[3]
        scale = tf.get_variable("scale", [depth], initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
        offset = tf.get_variable("offset", [depth], initializer=tf.constant_initializer(0.0))
        mean, variance = tf.nn.moments(input, axes=[1, 2], keep_dims=True)
        epsilon = 1e-5
        inv = tf.rsqrt(variance + epsilon)
        normalized = (input - mean) * inv
        return scale * normalized + offset


def residual_bn(x, ks, name):
    """
    original residual
    """
    ci = x.get_shape().as_list()[3]
    with tf.variable_scope(name):
        net = conv2d(x, ci, ks, 1, 'SAME')
        net = relu(bn(net, training=is_training))
        net = conv2d(net, ci, ks, 1, 'SAME')
        net = bn(net, training=is_training)
    return relu(net + x)


def residual_bn_pre(x, ks, name):
    """
    pre-activation residual
    """

    ci = x.get_shape().as_list()[3]
    with tf.variable_scope(name):
        net = relu(bn(x, training=is_training))
        net = conv2d(net, ci, ks, 1, 'SAME')
        net = relu(bn(net, training=is_training))
        net = conv2d(net, ci, ks, 1, 'SAME')
    return x + net


# 可以设置stride同时使用instance_norm
def residual_in(x, ks=3, s=1, name='res'):
    ci = x.get_shape().as_list()[3]
    p = int((ks - 1) / 2)
    with tf.variable_scope(name):
        y = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
        y = relu(instance_norm(conv2d(y, ci, ks, s, 'VALID'), 'in1'))
        y = tf.pad(y, [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
        y = instance_norm(conv2d(y, ci, ks, s, 'VALID'), 'in2')
    return y + x
