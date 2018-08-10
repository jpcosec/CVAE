"""
Low level and deprecated methods
"""
import numpy as np
import tensorflow as tf


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

    def fc_initializer(self, input_channels, dtype=tf.float32):
        def _initializer(shape, dtype=dtype, partition_info=None):
            d = 1.0 / np.sqrt(input_channels)
            return tf.random_uniform(shape, minval=-d, maxval=d)

        return _initializer

    # Crea pesos y grafo para capas FC
    def _fc_weight_variable(self, weight_shape, name):
        name_w = "W_{0}".format(name)
        name_b = "b_{0}".format(name)

        input_channels = weight_shape[0]
        output_channels = weight_shape[1]
        d = 1.0 / np.sqrt(input_channels)
        bias_shape = [output_channels]

        weight = tf.get_variable(name_w, weight_shape, initializer=self.fc_initializer(input_channels))
        bias = tf.get_variable(name_b, bias_shape, initializer=self.fc_initializer(input_channels))
        return weight, bias

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
            weight_mean, bias_mean = self._fc_weight_variable([self.input_size, self.units], "z_mean")
            weight_sigma, bias_sigma = self._fc_weight_variable([self.input_size, self.units], "z_sigma")
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
