
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
import numpy as np

class Feature_Extractor(object):
    def __init__(self,
                 path_arr,
                 layer="block2_conv2",
                 pres_model=VGG19):

        self.base_model = pres_model(weights='imagenet')
        self.layer = layer
        self.model = Model(inputs=self.base_model.input, outputs=self.base_model.get_layer(self.layer).output)
        self.paths = path_arr

        self.fmaps = []
        # todo Dejar en un for por si acaso para despues
        for img_path in self.paths:
            img = image.load_img(img_path)
            img = image.img_to_array(img)

            f_map = self.map_prepros(img, self.prepros)
            self.fmaps.append(f_map)



    def print_layers(self):
        global base_model
        for layer in base_model.layers:
            print(layer.name)


    def prepros(self,x):
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        return self.model.predict(x)


    def calcular_pad(self,shape, dims=(224, 224), centrado=True):
        """
          Calcula el pad centrado
        """
        # Se obtiene cantidad de cortes y los residuos
        steps = [shape[0] // dims[0], shape[1] // dims[1]]
        remainders = (shape[0] % dims[0], shape[1] % dims[1])
        pads = []

        # para cada dimension se obtiene el padding necesario
        for i in np.arange(2):
            # Se calcula los pads que faltan
            total_pad = dims[i] - remainders[i]
            # si no faltaba nada se agrega un pad de o
            if remainders[i] == 0:
                pad = (0, 0)
            # Si no se necesita centrado
            elif not centrado:
                # Se usa pad total
                pad = (0, int(total_pad))
            # Si es centrado
            else:
                # si es par se calcula equitativamente
                if total_pad % 2 == 0:
                    steps[i] += 1
                    pad = (total_pad // 2, total_pad // 2)
                # Si no se calcula dejando uno extra a la derecha
                else:
                    steps[i] += 1
                    pad = (total_pad // 2, total_pad // 2 + 1)
            pads.append(pad)

        return pads, steps


    def map_prepros(self,img, prep_fun, dims=(224, 224)):
        # Se calculan pads
        pads, steps = self.calcular_pad(img.shape, dims)
        # Se realiza zero padding para "cuadrar" cortes
        img = np.pad(img, (pads[0], pads[1], (0, 0)), 'constant')

        # Se realizan cortes horizontales
        hcuts = np.hsplit(img, steps[1])
        hfeatures = []
        for hcut in hcuts:
            # Sobre los que se realizan cortes verticales
            vcuts = np.vsplit(hcut, steps[0])
            vfeatures = []
            for cut in vcuts:
                # Sobre los que se extraen las caracteristicas
                features = prep_fun(cut)
                vfeatures.append(features)
            # Para finalmente concatenarlas horizontal
            vmap = np.concatenate(vfeatures, axis=1)
            hfeatures.append(vmap)
        # Y verticalmente
        feature_map = np.concatenate(hfeatures, axis=2)
        # Para retornar un tensor de dimensiones de imagen paddeada y reducida por
        # Red de extraccion
        print("Dimensiones tensor de features", feature_map.shape)
        return feature_map


