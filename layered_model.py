import numpy as np
import tensorflow as tf

from model_utils import z_layer


# todo GENERAL- Dejar como VAE, probar con CIFAR, Usar en otro documento
# todo dar forma facil de obtener embeddings


# todo sacar el muestreo Z a una funcion externa (o modulo)
class CVAE(object):
    """ Based on Beta Variational Auto Encoder, V1 """

    # Variacional
    def __init__(self,  # todo agregar winsize, stride,
                 im_size,
                 gamma=0,  # Peso de z encoding
                 capacity_limit=25.0,  # limite de capacidad
                 capacity_change_duration=100000,
                 learning_rate=5e-4,  # achicar a 16,16
                 winsize=(16, 16),
                 in_channel=128,
                 triplet_gain=0.6):  # peso de triplett

        self.winsize = winsize
        self.in_channel = in_channel
        self.flat_size = self.winsize[0] * self.winsize[1] * self.in_channel
        self.batch_shape = [-1, self.winsize[0], self.winsize[1], self.in_channel]

        # Cosas que van a servir
        self.win_area = self.winsize[0] * self.winsize[1]
        self.im_size = im_size
        self.im_diag = (self.im_size[0] * self.im_size[0]) + (self.im_size[1] * self.im_size[1])

        self.gamma = gamma
        self.capacity_limit = capacity_limit
        self.capacity_change_duration = capacity_change_duration
        self.learning_rate = learning_rate

        # Create siamese autoencoder
        self._create_siamese_network()

        # Define loss function and corresponding optimizer
        self.triplet_gain = triplet_gain
        self._create_loss_optimizer()

    # Crea red encoder parcial
    def _create_encoder_network(self, x, reuse=False):
        with tf.variable_scope("encoder", reuse=reuse) as scope:
            # [filter_height, filter_width, in_channels, out_channels]
            # with tf.variable_scope("c1")

            x_reshaped = tf.reshape(x, self.batch_shape)  # [-1,16,16,3]
            tf.summary.image('input', x_reshaped, 1)

            # todo crear iterativamente
            net = tf.layers.conv2d(x_reshaped,  # [-1,8,8,32]
                                   filters=32,  # todo Variabilizar n_channels
                                   kernel_size=(4, 4),  # todo_variabilizar canales
                                   strides=(2, 2),
                                   activation=tf.nn.leaky_relu,
                                   padding="same",
                                   name="conv1")

            net = tf.layers.conv2d(net,  # [-1,4,4]
                                   filters=32,  # todo Variabilizar n_channels
                                   kernel_size=(4, 4),  # todo_variabilizar canales
                                   strides=(2, 2),
                                   activation=tf.nn.leaky_relu,
                                   padding="same",
                                   name="conv2")
            """net = tf.layers.conv2d(net,
                                   filters=32,  # todo Variabilizar n_channels
                                   kernel_size=(4, 4),
                                   strides=(2, 2),
                                   activation=tf.nn.leaky_relu,
                                   padding="same",
                                   name="conv3")
            net = tf.layers.conv2d(net,
                                   filters=32,  # todo Variabilizar n_channels
                                   kernel_size=(4, 4),  # todo_variabilizar canales
                                   strides=(2, 2),
                                   activation=tf.nn.leaky_relu,
                                   padding="same",
                                   name="conv4")"""

            net = tf.reshape(net, [-1, 4 * 4 * 32])
            net = tf.layers.dense(net,
                                  units=4 * 4 * 32,
                                  activation=tf.nn.relu,
                                  name="dense1")
            net = tf.layers.dense(net,
                                  units=256,
                                  activation=tf.nn.relu,
                                  name='dense2')

            return net

    # Crea grafo de decoder
    def _create_decoder_network(self, input, reuse=False):
        with tf.variable_scope("decoder", reuse=reuse) as scope:
            net = tf.layers.dense(input,
                                  units=256,
                                  activation=tf.nn.relu,
                                  name='dense2')
            net = tf.layers.dense(net,
                                  units=4 * 4 * 32,
                                  activation=tf.nn.relu,
                                  name="dense1")

            net = tf.reshape(net, [-1, 4, 4, 32])  # 4x4

            net = tf.layers.conv2d_transpose(net,  # 8x8
                                             filters=32,
                                             kernel_size=(4, 4),
                                             strides=(2, 2),
                                             padding='same',
                                             activation=tf.nn.leaky_relu,
                                             name="deconv1")
            net = tf.layers.conv2d_transpose(net,  # 16x16
                                             filters=self.in_channel,
                                             kernel_size=(4, 4),
                                             strides=(2, 2),
                                             padding='same',

                                             activation=tf.nn.leaky_relu,
                                             name="deconv2")
            """net = tf.layers.conv2d_transpose(net,
                                             filters=32,
                                             kernel_size=(4, 4),
                                             strides=(2, 2),
                                             padding='same',
                                             activation=tf.nn.leaky_relu,
                                             name="deconv3")
            net = tf.layers.conv2d_transpose(net,
                                             filters=self.inchannel,
                                             kernel_size=(4, 4),
                                             strides=(2, 2),
                                             padding='same',
                                             activation=tf.nn.leaky_relu,
                                             name="deconv4")"""

            x_out_logit = tf.reshape(net, [-1, self.flat_size])

            tf.summary.image('output', net, 1)

            return x_out_logit

    # Crea grafo de red completa
    def _create_vae_network(self, x, capacity):

        with tf.variable_scope("B-Vae"):
            encoder = self._create_encoder_network(x)  # todo Separar dentro de red siamesa

            z = z_layer(encoder, 256, 32, name="Embedding", gamma=self.gamma)
            # z_sample = z._sample_z()
            """self.embedding_input = z.mean()
            self.embedding_size = 32"""
            tf.summary.histogram("z_mean", z.mean())
            tf.summary.histogram("z_variance", z.variance())

            x_out_logit = self._create_decoder_network(z.sample_z())
            x_out = tf.nn.sigmoid(x_out_logit)

            # info de salida
            net = {'x': x,
                   'o': x_out_logit,
                   'z_variance': z.variance(),
                   'z_mean': z.mean(),
                   'loss': z.net_loss(capacity=capacity, label=x, logits=x_out),
                   'rec_loss': z.reconstr_loss,
                   'latent_loss': z.latent_loss}

            return net

    # Crea red siamesa completa
    def _create_siamese_network(self):  # todo desacoplar perdida no vae

        # Placeholders
        self.im = tf.placeholder(tf.float32, shape=[None, self.winsize[0] * self.winsize[1] * self.in_channel],
                                 name="input1")
        self.posim = tf.placeholder(tf.float32, shape=[None, self.winsize[0] * self.winsize[1] * self.in_channel],
                                    name="input_pos")
        self.negim = tf.placeholder(tf.float32, shape=[None, self.winsize[0] * self.winsize[1] * self.in_channel],
                                    name="input_neg")
        # Encoding capacity
        self.c_anchor = tf.placeholder(tf.float32, shape=[], name="capacity_anchor")
        self.c_pos = tf.placeholder(tf.float32, shape=[], name="capacity_positive")
        self.c_neg = tf.placeholder(tf.float32, shape=[], name="capacity_negative")

        with tf.variable_scope("siamese") as scope:
            self.anchornet = self._create_vae_network(self.im, self.c_anchor)
            scope.reuse_variables()
            self.posnet = self._create_vae_network(self.posim, self.c_pos)
            self.negnet = self._create_vae_network(self.negim, self.c_neg)

    # Crea perdida de diferencia
    def _triplet_loss(self, margin):
        with tf.variable_scope("triplet_loss") as scope:
            # Se calcula distancia cuadratica
            d_pos = tf.reduce_sum(tf.square(self.anchornet['z_mean'] - self.posnet['z_mean']), 1)
            d_neg = tf.reduce_sum(tf.square(self.anchornet['z_mean'] - self.negnet['z_mean']), 1)

            self.triplet_loss = tf.maximum(0., margin + d_pos - d_neg)
            self.triplet_loss = tf.reduce_mean(self.triplet_loss)
            triplet_loss_summary_op = tf.summary.scalar('triplet_loss', self.triplet_loss)

            return self.triplet_loss

    # Crea grafo de funcion de perdida
    def _create_loss_optimizer(self):
        with tf.variable_scope("total_loss") as scope:
            self.triplet_margin = tf.placeholder(tf.float32, shape=[], name="triplet_margin")

            self.loss = self.anchornet['loss'] + (self.triplet_gain * self._triplet_loss(self.triplet_margin))

            self.summary_op = tf.summary.merge_all()

            self.optimizer = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate).minimize(self.loss)

    # Calcula dinamicamente la capacidad
    def _calc_encoding_capacity(self, step):
        if step > self.capacity_change_duration:
            c = self.capacity_limit
        else:
            c = self.capacity_limit * (step / self.capacity_change_duration)
        return c

    # Entrenamiento de mini-batch, retorna perdida
    def partial_fit(self, sess, images, pos, neg, step):
        """Train model based on mini-batch of input data.

        Return loss of mini-batch.
        """
        c = self._calc_encoding_capacity(step)
        margin = 0.1
        _, reconstr_loss, latent_loss, zmean, triplet_loss, summary_str = sess.run((self.optimizer,  # outs
                                                                                    self.anchornet['rec_loss'],
                                                                                    self.anchornet['latent_loss'],
                                                                                    self.anchornet['z_mean'],
                                                                                    self.triplet_loss,
                                                                                    self.summary_op),
                                                                                   feed_dict={  # ins todo agregar weas
                                                                                       self.im: images,
                                                                                       self.posim: pos,
                                                                                       self.negim: neg,
                                                                                       self.c_anchor: c,
                                                                                       self.c_pos: c,
                                                                                       self.c_neg: c,
                                                                                       self.triplet_margin: margin
                                                                                   })
        return reconstr_loss, latent_loss, zmean, triplet_loss, summary_str


"""
    # Reconstruye la data dada
    def reconstruct(self, sess, xs):
        # Reconstruct given data. 
        # Original VAE output
        return sess.run(self.x_out,
                        feed_dict={self.x: xs})

    # Mapea la data al espacio latente
    def transform(self, sess, xs):
        #Transform data by mapping it into the latent space.
        return sess.run([self.z_mean, self.z_log_sigma_sq],
                        feed_dict={self.x: xs})

    # Genera imagen desde espacio latente
    def generate(self, sess, zs):
        # Generate data by sampling from latent space.
        return sess.run(self.x_out,
                        feed_dict={self.z: zs})
"""
