import tensorflow as tf
from keras_vggface.vggface import VGGFace

from model import hparams
from model.celeb_similarity.model.model_output import ModelOutput
from model.celeb_similarity.predictions_decoder import PredictionsDecoder


class Inference:
    def __init__(self, model_name='resnet50'):
        self._session = tf.Session()
        self._graph = tf.get_default_graph()
        self._model = self._load_model(model_name)
        self._predictions_decoder = PredictionsDecoder(model_name)

    def run(self, img) -> [ModelOutput]:
        # start = time.time()
        predictions = self._predict(img)
        decoded_predictions = self._predictions_decoder.decode_predictions(predictions)
        predictions_num = len(decoded_predictions)
        domain_model_outputs = [ModelOutput.from_raw_model_output(decoded_predictions[i]) for i in
                                range(predictions_num)]
        # t = time.time() - start
        # print(t * 1000, 'ms')

        return domain_model_outputs

    def _load_model(self, model_name):
        with self._graph.as_default():
            with self._session.as_default():
                vgg_face = VGGFace(model=model_name, input_shape=hparams.feed_img_size)
        return vgg_face

    def _predict(self, img):
        with self._graph.as_default():
            with self._session.as_default():
                predictions = self._model.run(img)
        return predictions
