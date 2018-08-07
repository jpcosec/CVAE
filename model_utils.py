"""
Low level and deprecated methods
"""
import numpy as np
import tensorflow as tf


def fc_initializer(input_channels, dtype=tf.float32):
    def _initializer(shape, dtype=dtype, partition_info=None):
        d = 1.0 / np.sqrt(input_channels)
        return tf.random_uniform(shape, minval=-d, maxval=d)

    return _initializer


# Inicializador para convolucional
def conv_initializer(kernel_width, kernel_height, input_channels, dtype=tf.float32):
    def _initializer(shape, dtype=dtype, partition_info=None):
        d = 1.0 / np.sqrt(input_channels * kernel_width * kernel_height)
        return tf.random_uniform(shape, minval=-d, maxval=d)

    return _initializer

    # Funciones auxiliares


# Crea pesos y grafo para capas convolucionales y deconvolucionales
def _conv2d_weight_variable(weight_shape, name, deconv=False):
    name_w = "W_{0}".format(name)
    name_b = "b_{0}".format(name)

    w = weight_shape[0]
    h = weight_shape[1]
    if deconv:
        input_channels = weight_shape[3]
        output_channels = weight_shape[2]
    else:
        input_channels = weight_shape[2]
        output_channels = weight_shape[3]
    d = 1.0 / np.sqrt(input_channels * w * h)
    bias_shape = [output_channels]

    weight = tf.get_variable(name_w, weight_shape,
                             initializer=conv_initializer(w, h, input_channels))
    bias = tf.get_variable(name_b, bias_shape,
                           initializer=conv_initializer(w, h, input_channels))
    return weight, bias


# Crea pesos y grafo para capas FC
def _fc_weight_variable(weight_shape, name):
    name_w = "W_{0}".format(name)
    name_b = "b_{0}".format(name)

    input_channels = weight_shape[0]
    output_channels = weight_shape[1]
    d = 1.0 / np.sqrt(input_channels)
    bias_shape = [output_channels]

    weight = tf.get_variable(name_w, weight_shape, initializer=fc_initializer(input_channels))
    bias = tf.get_variable(name_b, bias_shape, initializer=fc_initializer(input_channels))
    return weight, bias


# Obtiene porte salida para  deconvoluciones
def _get_deconv2d_output_size(input_height, input_width, filter_height,
                              filter_width, row_stride, col_stride, padding_type):
    if padding_type == 'VALID':
        out_height = (input_height - 1) * row_stride + filter_height
        out_width = (input_width - 1) * col_stride + filter_width
    elif padding_type == 'SAME':
        out_height = input_height * row_stride
        out_width = input_width * col_stride
    return out_height, out_width


# Crea capa convolucional 2d
def _conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1],
                        padding='SAME')


# Crea capa deconvolucional
def _deconv2d(x, W, input_width, input_height, stride):
    filter_height = W.get_shape()[0].value
    filter_width = W.get_shape()[1].value
    out_channel = W.get_shape()[2].value

    out_height, out_width = _get_deconv2d_output_size(input_height,
                                                      input_width,
                                                      filter_height,
                                                      filter_width,
                                                      stride,
                                                      stride,
                                                      'SAME')
    batch_size = tf.shape(x)[0]

    output_shape = tf.stack([batch_size, out_height, out_width, out_channel])
    return tf.nn.conv2d_transpose(x, W, output_shape,
                                  strides=[1, stride, stride, 1],
                                  padding='SAME')


def deconv2d_relu(input, name, shape=(4, 4, 32, 32), input_shape=(4, 4), stride=2):
    with tf.variable_scope(name) as scope:
        # [filter_height, filter_width, output_channels, in_channels]
        weight, bias = _conv2d_weight_variable(shape, name, deconv=True)
        return tf.nn.relu(_deconv2d(input, weight, input_shape[0], input_shape[1], stride) + bias)


def conv2d_relu(input, name, shape=(4, 4, 32, 32)):
    with tf.variable_scope(name):
        weight, bias = _conv2d_weight_variable(shape, name)
        return tf.nn.relu(_conv2d(input, weight, 2) + bias)  # (32, 32)


def FC_relu(input, name, shape=(4 * 4 * 32, 256), reshape=0):
    with tf.variable_scope(name):
        if (reshape != 0):
            input = tf.reshape(input, (-1, reshape))
        weight, bias = _fc_weight_variable(shape, name)
        return tf.nn.relu(tf.matmul(input, weight) + bias)


class z_layer(object):
    def __init__(self,
                 input,
                 input_size,
                 units,
                 gamma,
                 name):
        self.input = input
        self.input_size = input_size
        self.units = units
        self.name = name
        self._create_net()
        self.gamma = gamma


    #def __call__(self, net):
    #    return self.sample_z()

    #def sample(self):
    #    return self.sample_z()

    # Obtiene sample de z
    def sample_z(self):
        with tf.variable_scope("sample_Z") as scope:
            eps_shape = tf.shape(self.z_mean)
            eps = tf.random_normal(eps_shape, 0, 1, dtype=tf.float32)
            # z = mu + sigma * epsilon
            return tf.add(self.z_mean, tf.multiply(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))

    def mean(self):
        return self.z_mean

    def variance(self):
        return self.z_log_sigma_sq

    def _create_net(self):
        with tf.variable_scope(self.name, ):
            weight_mean, bias_mean = _fc_weight_variable([self.input_size, self.units], "z_mean")
            weight_sigma, bias_sigma = _fc_weight_variable([self.input_size, self.units], "z_sigma")
            self.z_mean = tf.matmul(self.input, weight_mean) + bias_mean
            self.z_log_sigma_sq = tf.matmul(self.input, weight_sigma) + bias_sigma

    def get_latent_loss(self):
        return self.latent_loss

    def get_reconstruction_loss(self):
        return self.reconstr_loss

    def net_loss(self, capacity, logits, label):
        with tf.variable_scope("VAE_loss"):
            with tf.variable_scope("reconstruction_loss"):
                # Reconstruction loss
                reconstr_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=logits)
                reconstr_loss = tf.reduce_sum(reconstr_loss, 1)
                self.reconstr_loss = tf.reduce_mean(reconstr_loss)
                reconstr_loss_summary_op = tf.summary.scalar('reconstr_loss', self.reconstr_loss)

            with tf.variable_scope("latent_loss"):
                # Latent loss
                latent_loss = -0.5 * tf.reduce_sum(1 +self.variance()
                                                   - tf.square(self.mean())
                                                   - tf.exp(self.variance()), 1)
                self.latent_loss = tf.reduce_mean(latent_loss)
                latent_loss_summary_op = tf.summary.scalar('latent_loss', self.latent_loss)

            # Loss with encoding capacity term
            return self.reconstr_loss + self.gamma * tf.abs(self.latent_loss - capacity)
