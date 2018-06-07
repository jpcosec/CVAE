import tensorflow as tf
import math
from scipy import misc
import numpy as np


class DataManager(object):
    """" Reads a image and provides management"""

    def __init__(self,
                 filename):
        # todo modificar para guardar info de pos xy
        self.data = []
        self.n_samples = 0
        self.load(filename)

    @property
    def sample_size(self):
        return self.n_samples

    #todo modificar para enviar 17 cortes (
    def get_images(self, indices):
        images = []
        for index in indices:
            img = self.data[index]
            img = img.reshape(-1)
            images.append(img)
        return images

    #def get_random_images(self, size):
    #    indices = [np.random.randint(self.n_samples) for i in range(size)]
    #    return self.data[indices]

    # Crops a loaded image, the stride is adjusted for remainders
    def get_crops(self, window_size, stride):
        x_wins = int(math.floor((self.image_shape[0] - window_size[0]) / stride[0]) + 1)
        y_wins = int(math.floor((self.image_shape[1] - window_size[1]) / stride[1]) + 1)
        self.n_samples += x_wins * y_wins

        xtaps = self.taps(0, x_wins, window_size, stride)
        ytaps = self.taps(1, y_wins, window_size, stride)
        for i in np.arange(x_wins):
            xt = xtaps[i]
            for j in np.arange(y_wins):
                yt = ytaps[j]
                self.data.append(self.image[xt[0]:xt[1], yt[0]:yt[1]])

    def taps(self, dim, windows, size, stride):
        # todo ver si el nombre esta bien
        # recorre la imagen anexando
        tap = []
        for i in np.arange(windows):
            x_1 = i * stride[dim]
            x_2 = (x_1) + size[dim]

            if (x_2 > self.image_shape[dim]):
                x_2 = self.image_shape[dim]
                x_1 = self.image_shape[dim] - size[dim]
            tap.append((x_1, x_2))

        return tap

    #   Loads images from folder using scypi and gets crops (way easier than TF method)
    def load(self, filename, window_size=(64, 64), stride=(4, 4)):
        # todo Agregar data augmentations
        # Loads image
        self.image = misc.imread(filename)
        # gets image shape
        self.image_shape = self.image.shape
        # Gets crops
        self.get_crops(window_size, stride)



    def ImageCrops(self, filenames, window_size=(64, 64), stride=(32, 32)):
        with tf.variable_scope("Crops_Gen"):
            #  list of files to read
            filename_queue = tf.train.string_input_producer(filenames)
            reader = tf.WholeFileReader()
            key, value = reader.read(filename_queue)
            # todo implementar para que sepa si JPEG o PNG
            # use png or jpg decoder based on your files.

            image = tf.image.decode_jpeg(value)
            # image gets reshaped from (w,h,c) to (1,w,h,c) for the extracter
            img_reshaped = tf.reshape(image, [1, self.image_shape[0], self.image_shape[1], 3])

            x_wins = math.floor((self.image_shape[0] - window_size[0]) / stride[0]) + 1
            y_wins = math.floor((self.image_shape[1] - window_size[1]) / stride[1]) + 1

            self.n_samples = int(x_wins * y_wins)

            init_op = tf.global_variables_initializer()
            with tf.Session() as sess:
                sess.run(init_op)

                # Start populating the filename queue.

                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(coord=coord)

                for i in range(len(filenames)):  # length of your filename list

                    # Patches get extracted from images
                    patches = tf.extract_image_patches(img_reshaped,
                                                       [1, window_size[0], window_size[1], 1],  # window size
                                                       [1, stride[0], stride[1], 1],  # stride
                                                       [1, 1, 1, 1],  # rate (?)
                                                       padding='VALID',
                                                       name='Sliding_window').eval()

                    coord.request_stop()
                    coord.join(threads)
                # self.summary_op = tf.summary.merge_all()
                # genwriter = tf.summary.FileWriter("./log/gen")
                # tf.summary.image('crudo', img_reshaped, 3)
                # writer.add_graph(sess.graph)

                return patches
