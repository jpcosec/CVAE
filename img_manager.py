import tensorflow as tf
import math
from scipy import misc
from scipy import io
from scipy.stats import mode
import numpy as np
from Feature_Extractor import Feature_Extractor
from scipy.spatial import distance


# todo modificar para procesar varias imagenes
# todo calcular campo receptivo de labels

class DataManager(object):
    """" Reads a image and provides management"""

    def __init__(self,
                 filename,
                 window_size=(16, 16),
                 stride=(4, 4),
                 sprite_dir=None,
                 gt_dir=None,
                 format='mat'):
        # -----------
        # asignaciones externas
        # -----------
        self.stride = stride
        self.window_size = window_size

        # -----------
        # Inicializaciones vacias
        # -----------
        self.data = []
        self.labels = []
        self.xposs = []
        self.yposs = []
        self.n_samples = 0

        # -----------
        # Definiciones internas
        # -----------
        self.nmid = (int(window_size[0] / (stride[0] * 2) - 1), int(window_size[1] / (stride[1] * 2) - 1))

        # -----------
        #   Ejecuciones
        # ------------

        # Crea labels
        self.load_and_preprocess(filename)

        # Crea labels para hacer test
        if gt_dir != None:
            if format == 'mat':
                self.make_labels_from_mat(gt_dir)
            elif format == 'img':
                self.make_labels_from_img(gt_dir)

        # Crea sprites para embedding
        if sprite_dir != None:
            self.make_sprites(spritename=sprite_dir)



    # ---------
    # propiedades
    # ---------
    @property
    def sample_size(self):
        return self.n_samples

    @property
    def get_im_size(self):
        return self.img_features_shape[0], self.img_features_shape[1]

    # ---------
    # Procesamiento de imagenes
    # ---------
    def load_and_preprocess(self, filename):

        # todo Agregar data augmentations (Obstrucciones?, imagenes generadas con VGG)
        # Loads img_features
        # self.img_features = misc.imread(filename)
        self.FE = Feature_Extractor(path_arr=[filename])  # todo modificar cuando se incorpore VGG directo a la red
        self.img_features = self.FE.features[0][0, :, :, :]
        self.image_shape =  self.FE.image_shape
        self.image = self.FE.image

        self.img_features_shape = self.img_features.shape
        print(self.img_features_shape)
        # numero de ventanas
        self.ywins = int(math.floor((self.img_features_shape[1] - self.window_size[1]) // self.stride[1]) + 1)
        self.xwins = int(math.floor((self.img_features_shape[0] - self.window_size[0]) // self.stride[0]) + 1)
        # Gets crops
        self.get_crops()

    # Devuelve vector con detenciones de la imagen
    def taps(self, dim, windows, size, stride):
        # cambiar nombre
        # recorre la imagen anexando
        tap = []
        for i in np.arange(windows):
            x_1 = i * stride[dim]
            x_2 = (x_1) + size[dim]

            if (x_2 > self.img_features_shape[dim]):
                x_2 = self.img_features_shape[dim]
                x_1 = self.img_features_shape[dim] - size[dim]
            tap.append((x_1, x_2))

        return tap

    # Crops a loaded img_features, the stride is adjusted for remainders
    def get_crops(self):
        self.n_samples += self.xwins * self.ywins

        xtaps = self.taps(0, self.xwins, self.window_size, self.stride)
        ytaps = self.taps(1, self.ywins, self.window_size, self.stride)
        for i in np.arange(self.xwins):
            xt = xtaps[i]
            for j in np.arange(self.ywins):
                yt = ytaps[j]
                self.data.append(self.img_features[xt[0]:xt[1], yt[0]:yt[1]])
                self.xposs.append(xt[0])
                self.yposs.append(yt[0])

    # Avisa si la imagen esta dentro del rango
    def en_rango(self, x, y):
        if x > self.xwins or y > self.ywins or x < 0 or y < 0:
            return False
        else:
            return True

    # Retorna booleano sobre si imagenes se parecen o no
    def valid_triplet(self, anchor, pos, neg, umbral=0.1):

        fx = anchor.flatten()
        fy = pos.flatten()
        fz = neg.flatten()

        dist_pos = distance.euclidean(fy,fx)
        dist_neg = distance.euclidean(fz,fx)

        return dist_pos/dist_neg


    # ---------
    # Triplet batch
    # ---------
    # Genera indices aleatorios de tripletas
    def rand_triplets(self, indices, margin=0.1):
        index_array = []

        for index in indices: #probar triplet
            # se encuentra ejemplo positivo ventana dentro de iou primero
            while (False):
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
                    if abs(self.xposs[negid] - self.xposs[index]) > margin * self.img_features_shape[0]:
                        if abs(self.yposs[negid] - self.yposs[index]) > margin * self.img_features_shape[1]:
                            break

                if self.valid_triplet(self.data[index], self.data[posid], self.data[negid]):
                    break

            index_array.append((posid, negid))

        return index_array

    # Retorna batch de triplets
    def get_batch(self, indices, margin=0.1):
        print("generando batch")
        images = []
        posimages = []
        negimages = []

        indexes = self.rand_triplets(indices, margin)
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

    # ---------
    # Labels
    # ---------

    # Retorna labels de indices
    def get_labels(self, indices):
        batch_labels = []
        for i in np.arange(len(indices)):
            batch_labels.append(self.labels[i])
        return batch_labels

    # Funcion para crear labels desde groundtruth en formato imagen
    def make_labels_from_img(self, ground_truth_path):
        """
        Crea labels de celdas desde pixel-wise label map
        :param ground_truth_path:
        :return:
        """
        print('creando labels desde', ground_truth_path)
        gt = misc.imread(ground_truth_path)

        self.make_labels(gt)

    # Funcion para crear labels desde groundtruth
    def make_labels_from_mat(self, ground_truth_path):

        """
        S: The pixel-wise label map of size [height x width].
        names: The names of the thing and stuff classes in COCO-Stuff. For more details see Label Names & Indices.
        captions: Image captions from [2] that are annotated by 5 distinct humans on average.
        regionMapStuff: A map of the same size as S that contains the indices for the approx. 1000 regions (superpixels) used to annotate the img_features.
        regionLabelsStuff: A list of the stuff labels for each superpixel. The indices in regionMapStuff correspond to the entries in regionLabelsStuff.
        """

        mat = io.loadmat(ground_truth_path)
        gt = mat["S"]

        self.make_labels(gt)

    # Funcion que calcula labels usando moda
    def make_labels(self, ground_truth):
        """
        Procesa ground_truth ya cargado y retorna labels de celdas

        :param ground_truth:
        :return:
        """

        self.ground_truth = self.FE.zero_pad(ground_truth)
        # Se realiza zero padding para "cuadrar" cortes

        # se corrige el stride con aquel generado por la reduccion dimensional del preprocesamiento
        prep_red = (
        self.image_shape[0] / self.img_features_shape[0], self.image_shape[1] / self.img_features_shape[1])
        stride = (self.stride[0] * prep_red[0], self.stride[1] * prep_red[1])
        window_size=(self.window_size[0]*prep_red[0],self.window_size[1]*prep_red[1])

        # Se obtienen detenciones
        xtaps = self.taps(0, self.xwins, window_size, stride)
        ytaps = self.taps(1, self.ywins, window_size, stride)

        # Se obtienen labels usando moda
        for i in np.arange(self.xwins):
            xt = xtaps[i]
            for j in np.arange(self.ywins):
                yt = ytaps[j]
                window = self.ground_truth[xt[0]:xt[1], yt[0]:yt[1]]
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

    """
    def get_images(self, indices):
        print("img_4")
        # batchs pares
        images = []

        for index in indices:
            img = self.data[index]
            img = img.reshape(-1)
            images.append(img)
        return images
funcion para recibir cortes

    # retorna imagen llamada por indice
    def get_im_by_pos(self, x, y):
        print("img_3")

        if x > self.xwins or y > self.ywins:
            print("fuera de rango")
        index = (self.xwins * x) + y
        return self.data[index].reshape(-1)

    # retorna tuplas de vecinos
    # todo deprecate
    def get_neighbour_tuples(self, x, y):
        print("img_2")
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
        print("img_1")
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
    # Retorna cortes de imagen y posicion x,y
    def get_dataset(self):
        print("img_13")
        # todo ver que pasa si se usa directamente la distancia en pixeles
        return self.data, self.xposs, self.yposs"""


if __name__ == '__main__':
    manager = DataManager(filename="test_IMG.jpg",
                          gt_dir="test_segm.mat")

