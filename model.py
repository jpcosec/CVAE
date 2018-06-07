import numpy as np
import tensorflow as tf


# todo - Cambiar a xavier (probar)
# Inicializador para capa fully connected
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

# asdasdasd

# todo renombrar
class CVAE(object):
    """ Based on Beta Variational Auto Encoder, V1 """

    # Variacional
    def __init__(self,
                 gamma=100.0,
                 capacity_limit=25.0,
                 capacity_change_duration=100000,
                 learning_rate=5e-4):
        self.gamma = gamma
        self.capacity_limit = capacity_limit
        self.capacity_change_duration = capacity_change_duration
        self.learning_rate = learning_rate

        # Create autoencoder network
        self._create_network()

        # Define loss function and corresponding optimizer
        self._create_loss_optimizer()
    # Funciones auxiliares

    # Crea pesos y grafo para capas convolucionales y deconvolucionales
    def _conv2d_weight_variable(self, weight_shape, name, deconv=False):
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
    def _fc_weight_variable(self, weight_shape, name):
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
    def _get_deconv2d_output_size(self, input_height, input_width, filter_height,
                                  filter_width, row_stride, col_stride, padding_type):
        if padding_type == 'VALID':
            out_height = (input_height - 1) * row_stride + filter_height
            out_width = (input_width - 1) * col_stride + filter_width
        elif padding_type == 'SAME':
            out_height = input_height * row_stride
            out_width = input_width * col_stride
        return out_height, out_width
    # Crea capa convolucional 2d
    def _conv2d(self, x, W, stride):
        return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1],
                            padding='SAME')

    # Crea capa deconvolucional
    def _deconv2d(self, x, W, input_width, input_height, stride):
        filter_height = W.get_shape()[0].value
        filter_width = W.get_shape()[1].value
        out_channel = W.get_shape()[2].value

        out_height, out_width = self._get_deconv2d_output_size(input_height,
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

    def deconv2d_relu(self, input, name, shape=(4, 4, 32, 32), input_shape=(4, 4), stride=2):
        with tf.variable_scope(name) as scope:
            # [filter_height, filter_width, output_channels, in_channels]
            weight, bias = self._conv2d_weight_variable(shape, name, deconv=True)
            return tf.nn.relu(self._deconv2d(input, weight, input_shape[0], input_shape[1], stride) + bias)

    # Obtiene sample de z
    def _sample_z(self, z_mean, z_log_sigma_sq):
        with tf.variable_scope("sample_Z") as scope:
            eps_shape = tf.shape(z_mean)
            eps = tf.random_normal(eps_shape, 0, 1, dtype=tf.float32)
            # z = mu + sigma * epsilon
            z = tf.add(z_mean,
                       tf.multiply(tf.sqrt(tf.exp(z_log_sigma_sq)), eps))
            return z

    def conv2d_relu(self, input, name, shape=(4, 4, 32, 32)):
        with tf.variable_scope(name):
            weight, bias = self._conv2d_weight_variable(shape, name)
            return tf.nn.relu(self._conv2d(input, weight, 2) + bias)  # (32, 32)

    def FC_relu(self, input, name, shape=(4 * 4 * 32, 256), reshape=0):
        with tf.variable_scope(name):
            if (reshape != 0):
                input = tf.reshape(input, (-1, reshape))
            weight, bias = self._fc_weight_variable(shape, name)
            return tf.nn.relu(tf.matmul(input, weight) + bias)

    # Crea red encoder parcial
    def _create_encoder_network(self, x, reuse=False):
        with tf.variable_scope("encoder", reuse=reuse) as scope:
            # [filter_height, filter_width, in_channels, out_channels]
            # with tf.variable_scope("c1")

            x_reshaped = tf.reshape(x, [-1, 64, 64, 3])
            tf.summary.image('input', x_reshaped, 3)

            conv1 = self.conv2d_relu(x_reshaped, "conv1", (4, 4, 3, 32))  # (32, 32)
            conv2 = self.conv2d_relu(conv1, "conv2")  # (32, 32)
            conv3 = self.conv2d_relu(conv2, "conv3")  # (32, 32)
            conv4 = self.conv2d_relu(conv3, "conv4")  # (32, 32)

            fc1 = self.FC_relu(conv4, "fc1", reshape=4 * 4 * 32)
            fc2 = self.FC_relu(fc1, "fc2", shape=[256, 256])

            return fc2

    # Crea Z (variable latente)
    def _create_z_network(self, encoder, reuse=False):
        with tf.variable_scope("V_latente", reuse=reuse) as scope:
            weight_mean, bias_mean = self._fc_weight_variable([256, 32], "z_mean")
            weight_sigma, bias_sigma = self._fc_weight_variable([256, 32], "z_sigma")
            z_mean = tf.matmul(encoder, weight_mean) + bias_mean
            z_log_sigma_sq = tf.matmul(encoder, weight_sigma) + bias_sigma
            tf.summary.histogram("z_mean", z_mean)
            tf.summary.histogram("z_sigma", z_log_sigma_sq)

        return (z_mean, z_log_sigma_sq)

    # Crea grafo de decoder
    def _create_decoder_network(self, z, reuse=False):
        with tf.variable_scope("decoder", reuse=reuse) as scope:
            fc1 = self.FC_relu(z, "fc1", shape=[32, 256])
            fc2 = self.FC_relu(fc1, "fc2", shape=[256, 4 * 4 * 32])
            fc2_reshaped = tf.reshape(fc2, [-1, 4, 4, 32])

            deconv1 = self.deconv2d_relu(fc2_reshaped, name="deconv1", shape=(4, 4, 32, 32), input_shape=(4, 4),
                                         stride=2)
            deconv2 = self.deconv2d_relu(deconv1, name="deconv2", shape=(4, 4, 32, 32), input_shape=(8, 8),
                                         stride=2)
            deconv3 = self.deconv2d_relu(deconv2, name="deconv3", shape=(4, 4, 32, 32), input_shape=(16, 16),
                                         stride=2)
            # Deconvolucion de salida
            deconv4 = self.deconv2d_relu(deconv3, name="deconv4", shape=(4, 4, 3, 32), input_shape=(32, 32),
                                         stride=2)
            #  64 * 64 * 3 dimensiones de salida=entrada (64*64*3
            x_out_logit = tf.reshape(deconv4, [-1, 64 * 64 * 3])
            tf.summary.image('output', deconv4, 3)

            return x_out_logit

    # Crea grafo de red completa
    def _create_network(self):
        # tf Graph input
        self.x = tf.placeholder(tf.float32, shape=[None, 64 * 64 * 3], name="x")

        with tf.variable_scope("B-Vae"):
            self.h_fc2 = self._create_encoder_network(self.x)
            self.z_mean, self.z_log_sigma_sq = self._create_z_network(self.h_fc2)
            # Draw one sample z from Gaussian distribution
            # z = mu + sigma * epsilon
            self.z = self._sample_z(self.z_mean, self.z_log_sigma_sq)
            self.x_out_logit = self._create_decoder_network(self.z)
            self.x_out = tf.nn.sigmoid(self.x_out_logit)

    # Crea grafo de funcion de perdida
    def _create_loss_optimizer(self):
        with tf.variable_scope("loss"):
            with tf.variable_scope("reconstruction_loss"):
                # Reconstruction loss
                reconstr_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.x,
                                                                        logits=self.x_out_logit)
                reconstr_loss = tf.reduce_sum(reconstr_loss, 1)
                self.reconstr_loss = tf.reduce_mean(reconstr_loss)
                reconstr_loss_summary_op = tf.summary.scalar('reconstr_loss', self.reconstr_loss)

            with tf.variable_scope("latent_loss"):
                # Latent loss
                latent_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq
                                                   - tf.square(self.z_mean)
                                                   - tf.exp(self.z_log_sigma_sq), 1)
                self.latent_loss = tf.reduce_mean(latent_loss)
                latent_loss_summary_op = tf.summary.scalar('latent_loss', self.latent_loss)

            # todo agregar perdida por orden (triplet loss

            # Encoding capacity
            self.capacity = tf.placeholder(tf.float32, shape=[], name="capacity")
            # Loss with encoding capacity term
            self.loss = self.reconstr_loss + self.gamma * tf.abs(self.latent_loss - self.capacity)

            self.summary_op = tf.summary.merge_all()

            self.optimizer = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate).minimize(self.loss)

    # Calcula dinamicamente la capacidad
    # todo - poner como opcion
    # todo agregar summary
    def _calc_encoding_capacity(self, step):
        if step > self.capacity_change_duration:
            c = self.capacity_limit
        else:
            c = self.capacity_limit * (step / self.capacity_change_duration)
        return c

    # Entrenamiento de mini-batch, retorna perdida
    def partial_fit(self, sess, xs, step):
        """Train model based on mini-batch of input data.

        Return loss of mini-batch.
        """
        c = self._calc_encoding_capacity(step)
        _, reconstr_loss, latent_loss, summary_str = sess.run((self.optimizer,
                                                               self.reconstr_loss,
                                                               self.latent_loss,
                                                               self.summary_op),
                                                              feed_dict={
                                                                  self.x: xs,
                                                                  self.capacity: c
                                                              })
        return reconstr_loss, latent_loss, summary_str

    # Reconstruye la data dada
    def reconstruct(self, sess, xs):
        """ Reconstruct given data. """
        # Original VAE output
        return sess.run(self.x_out,
                        feed_dict={self.x: xs})

    # Mapea la data al espacio latente
    def transform(self, sess, xs):
        """Transform data by mapping it into the latent space."""
        return sess.run([self.z_mean, self.z_log_sigma_sq],
                        feed_dict={self.x: xs})

    # Genera imagen desde espacio latente
    def generate(self, sess, zs):
        """ Generate data by sampling from latent space. """
        return sess.run(self.x_out,
                        feed_dict={self.z: zs})
