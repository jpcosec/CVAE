import tensorflow as tf
import math
from scipy import misc
from scipy.stats import mode
import numpy as np
from Feature_Extractor import Feature_Extractor

# todo usar keras para preprocesar imagen


class DataManager(object):
    """" Reads a image and provides management"""

    def __init__(self,
                 filename,
                 window_size=(16, 16),
                 stride=(4, 4),
                 sprite_dir=None,
                 gt_dir=None):

        self.window_size = window_size
        self.stride = stride
        self.data = []
        self.labels = []
        self.xposs = []
        self.yposs = []
        self.n_samples = 0
        self.load(filename)

        if gt_dir != None:
            self.make_labels(gt_dir)
        if sprite_dir != None:
            self.make_sprites(spritename=sprite_dir)

        # Definiciones
        self.nmid = (int(window_size[0] / (stride[0] * 2) - 1), int(window_size[1] / (stride[1] * 2) - 1))

    @property
    def sample_size(self):
        return self.n_samples

    #   Loads images from folder using scypi and gets crops (way easier than TF method)

    def get_im_size(self):
        return self.image_shape[0], self.image_shape[1]

    def load(self, filename):

        # todo Agregar data augmentations
        # Loads image
        #self.image = misc.imread(filename)
        FE = Feature_Extractor([filename])
        self.image= FE.fmaps[0][0,:,:,:]
        # gets image shape

        self.image_shape = self.image.shape
        print(self.image_shape)
        # numero de ventanas
        self.ywins = int(math.floor((self.image_shape[1] - self.window_size[1]) // self.stride[1]) + 1)
        self.xwins = int(math.floor((self.image_shape[0] - self.window_size[0]) // self.stride[0]) + 1)
        # Gets crops
        self.get_crops()

    # Retorna cortes de imagen y posicion x,y
    def get_dataset(self):
        # todo ver que pasa si se usa directamente la distancia en pixeles
        return self.data, self.xposs, self.yposs

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

    # Crops a loaded image, the stride is adjusted for remainders
    def get_crops(self):
        self.n_samples += self.xwins * self.ywins

        xtaps = self.taps(0, self.xwins, self.window_size, self.stride)
        ytaps = self.taps(1, self.ywins, self.window_size, self.stride)
        for i in np.arange(self.xwins):
            xt = xtaps[i]
            for j in np.arange(self.ywins):
                yt = ytaps[j]
                self.data.append(self.image[xt[0]:xt[1], yt[0]:yt[1]])
                self.xposs.append(xt[0])
                self.yposs.append(yt[0])

    def en_rango(self, x, y):
        if x > self.xwins or y > self.ywins or x < 0 or y < 0:
            return False
        else:
            return True

    def rand_ids(self, indices, margin=0.1):
        index_array = []

        for index in indices:
            # se encuentra ejemplo positivo ventana dentro de iou primero
            while (True):
                x = self.xposs[index] / self.stride[0] + np.random.random_integers(-self.nmid[0], high=self.nmid[0])
                y = self.yposs[index] / self.stride[0] + np.random.random_integers(-self.nmid[1], high=self.nmid[1])
                # print self.en_rango(x,y)

                if self.en_rango(x, y):
                    posid = int((self.xwins * x) + y)
                    if posid < self.n_samples:
                        break

            # print ("posid",posid)

            # se encuentra ejemplo negativo fuera de iou despues
            while (True):
                negid = np.random.randint(self.n_samples)
                # print ("negid",negid)
                if abs(self.xposs[negid] - self.xposs[index]) > margin * self.image_shape[0]:
                    if abs(self.yposs[negid] - self.yposs[index]) > margin * self.image_shape[1]:
                        break

            index_array.append((posid, negid))

        return index_array

    def get_batch(self, indices, margin=0.1):

        images = []
        posimages = []
        negimages = []

        indexes = self.rand_ids(indices, margin)
        # print ("indexes", indexes)


        for i in np.arange(len(indices)):
            index = indices[i]
            posid = indexes[i][0]
            negid = indexes[i][1]
            try:
                images.append(self.data[index].reshape(-1).astype(float) / 255)
            except:
                print('anch', i)
            try:
                posimages.append(self.data[posid].reshape(-1).astype(float) / 255)
            except:
                print('pos', i)
                print ('id', posid)
            try:
                negimages.append(self.data[negid].reshape(-1).astype(float) / 255)
            except:
                print('neg', i)
                print ('id', negid)
                # print max(images[-1])

        return [images, posimages, negimages]

    def get_labels(self, indices):
        batch_labels = []
        for i in np.arange(len(indices)):
            batch_labels.append(self.labels[i])
        return batch_labels

    # todo funcion para sacarse clases --- COMPLETAR URGNENTE
    def make_labels(self, ground_truth):
        print('creando labels desde', ground_truth)
        gt = misc.imread(ground_truth)
        self.labels = []
        xtaps = self.taps(0, self.xwins, self.window_size, self.stride)
        ytaps = self.taps(1, self.ywins, self.window_size, self.stride)
        for i in np.arange(self.xwins):
            xt = xtaps[i]
            for j in np.arange(self.ywins):
                yt = ytaps[j]
                window = gt[xt[0]:xt[1], yt[0]:yt[1]]
                mod = mode(window.flatten())[0][0]
                self.labels.append(mod)

    # Para hacer sprites para visualizacion
    def make_sprites(self, spritename):
        print("haciendo SPRITES")
        # se obtiene el porte final de imagen
        col = int(math.sqrt(self.n_samples))
        shape = (col * self.window_size[0], col * self.window_size[1], 3)
        sprite_img = np.zeros(shape, dtype=np.uint8)

        count = 0
        for i in range(col):
            x1 = i * self.window_size[0]
            x2 = x1 + self.window_size[0]

            for j in range(col):
                y1 = j * self.window_size[1]
                y2 = y1 + self.window_size[1]

                sprite_img[x1:x2, y1:y2, :] = self.data[count]
                count += 1

        misc.imsave(spritename, sprite_img)
        print("imagen guardada")

    # todo Aleatorizar y entregar en batch con labels
    def get_images(self, indices):
        # batchs pares
        images = []

        for index in indices:
            img = self.data[index]
            img = img.reshape(-1)
            images.append(img)
        return images

    """funcion para recibir cortes"""

    # retorna imagen llamada por indice
    def get_im_by_pos(self, x, y):

        if x > self.xwins or y > self.ywins:
            print("fuera de rango")
        index = (self.xwins * x) + y
        return self.data[index].reshape(-1)

    # retorna tuplas de vecinos
    # todo deprecate
    def get_neighbour_tuples(self, x, y):
        neighs = []

        # Control de borde
        if x == 0:
            x += 1
        elif x == self.xwins:
            x -= 1
        if y == 0:
            y += 1
        elif y == self.ywins:
            y -= 1

        for i in [x - 1, x, x + 1]:
            for j in [y - 1, y, y + 1]:
                neighs.append((i, j))
        return neighs

    # retorna imagenes flattened desde el array pedido
    def get_neighbourhood(self, pos_arr):
        boxes = []
        for pos in pos_arr:
            box = []
            tups = self.get_neighbour_tuples(pos[0], pos[1])
            for (x, y) in tups:
                box.append(self.get_im_by_pos(x, y))
            boxes.append(box)
        return boxes

        # def get_random_images(self, size):
        #    indices = [np.random.randint(self.n_samples) for i in range(size)]
        #    return self.data[indices]
