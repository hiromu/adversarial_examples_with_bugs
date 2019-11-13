import math
import tensorflow as tf

slim = tf.contrib.slim

def _residual_block(X, nf_output, resample, bn_params, kernel_size=[3,3], name='res_block'):
    with tf.variable_scope(name):
        input_shape = X.shape
        nf_input = input_shape[-1]
        if resample == 'down': # Downsample
            shortcut = slim.avg_pool2d(X, [2,2])
            shortcut = slim.conv2d(shortcut, nf_output, kernel_size=[1,1], activation_fn=None) # init xavier

            net = slim.layer_norm(X, activation_fn=tf.nn.relu)
            net = slim.conv2d(net, nf_input, kernel_size=kernel_size, biases_initializer=None) # skip bias
            net = slim.layer_norm(net, activation_fn=tf.nn.relu)
            net = slim.conv2d(net, nf_output, kernel_size=kernel_size)
            net = slim.avg_pool2d(net, [2,2])

            return net + shortcut
        elif resample == 'up': # Upsample
            upsample_shape = map(lambda x: int(x)*2, input_shape[1:3])
            shortcut = tf.image.resize_nearest_neighbor(X, upsample_shape) 
            shortcut = slim.conv2d(shortcut, nf_output, kernel_size=[1,1], activation_fn=None)

            net = slim.batch_norm(X, activation_fn=tf.nn.relu, **bn_params)
            net = tf.image.resize_nearest_neighbor(net, upsample_shape) 
            net = slim.conv2d(net, nf_output, kernel_size=kernel_size, biases_initializer=None) # skip bias
            net = slim.batch_norm(net, activation_fn=tf.nn.relu, **bn_params)
            net = slim.conv2d(net, nf_output, kernel_size=kernel_size)

            return net + shortcut
        else:
            raise Exception('invalid resample value')


def generator(z, is_training=True, reuse=None, scope='Generator', fused_batch_norm=False, size=32):
    bn_params = {
        'is_training': is_training,
        'zero_debias_moving_mean': True,
        'fused': fused_batch_norm,
    }

    with tf.variable_scope(scope, reuse=reuse):
        net = slim.fully_connected(z, 8 * 8 * (size * 8), activation_fn=None)
        net = tf.reshape(net, [-1, 8, 8, size * 8])
        for i in xrange(1, int(math.log(size / 8, 2) + 1)):
            net = _residual_block(net, size * 8 / (2 ** i) , resample='up', bn_params=bn_params, name='res_block%d' % i)
        net = slim.conv2d(net, 3, kernel_size=[3,3], activation_fn=tf.nn.tanh)

        net.get_shape().assert_has_rank(4)
        net.get_shape().assert_is_compatible_with([None, size, size, 3])

        return net


def discriminator(X, is_training=True, reuse=None, scope='Discriminator', fused_batch_norm=False, size=32):
    bn_params = {
        'is_training': is_training,
        'zero_debias_moving_mean': True,
        'fused': fused_batch_norm,
    }

    with tf.variable_scope(scope, reuse=reuse):
        net = slim.conv2d(X, 8, [3, 3], activation_fn=None)
        for i in xrange(1, int(math.log(size / 8, 2) + 1)):
            net = _residual_block(net, 8 * (2 ** i), resample='down', bn_params=bn_params, name='res_block%d' % i)
        net = slim.flatten(net)
        net = slim.fully_connected(net, 1, activation_fn=None)

        net.get_shape().assert_has_rank(2)
        net.get_shape().assert_is_compatible_with([None, 1])

        return net
