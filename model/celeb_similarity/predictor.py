from model import hparams
from model.celeb_similarity.model_output import ModelOutput

import tensorflow as tf
from keras_vggface.vggface import VGGFace
from keras_vggface import utils


class Predictor:
    def __init__(self):
        self.__session = tf.Session()
        self.__graph = tf.get_default_graph()
        self.__model = self.__load_model()

    def __load_model(self):
        with self.__graph.as_default():
            with self.__session.as_default():
                vgg_face = VGGFace(input_shape=hparams.feed_img_size)
        return vgg_face

    def predict(self, img) -> [ModelOutput]:
        with self.__graph.as_default():
            with self.__session.as_default():
                predictions = self.__model.predict(img)
                decoded_predictions = utils.decode_predictions(predictions)
                predictions_num = len(decoded_predictions)

                domain_model_outputs = [ModelOutput.from_raw_model_output(decoded_predictions[i]) for i in range(predictions_num)]

        return domain_model_outputs
