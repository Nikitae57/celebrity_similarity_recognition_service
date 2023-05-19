import numpy as np
from keras.utils.data_utils import get_file

from model import hparams as hp
from model.celeb_similarity.model.model_output import ModelOutput

V1_LABELS_PATH = 'https://github.com/rcmalli/keras-vggface/releases/download/v2.0/rcmalli_vggface_labels_v1.npy'
V2_LABELS_PATH = 'https://github.com/rcmalli/keras-vggface/releases/download/v2.0/rcmalli_vggface_labels_v2.npy'

VGGFACE_DIR = 'models/vggface'


class PredictionsDecoder:
    def __init__(self, model_name: str):
        self._validate_model_name(model_name)
        self.labels = self._load_labels(model_name)

    def decode_predictions(self, predictions, n_top_predictions=5):
        if len(predictions.shape) != 2:
            raise ValueError('`decode_predictions` expects '
                             'a batch of predictions '
                             '(i.e. a 2D array of shape (samples, 2622)) for V1 or '
                             '(samples, 8631) for V2.'
                             'Found array with shape: ' + str(predictions.shape))
        results = []
        for pred in predictions:
            top_indices = pred.argsort()[-n_top_predictions:][::-1]
            result = [[str(self.labels[i]).strip(), pred[i]] for i in top_indices]
            result.sort(key=lambda x: x[1], reverse=True)
            results.append(result)

        return results

    @staticmethod
    def _validate_model_name(model_name: str):
        allowed_model_names = [hp.RESNET_MODEL_NAME, hp.VGG_MODEL_NAME]
        if model_name not in allowed_model_names:
            raise ValueError(f'Invalid model name. Valid names are: {allowed_model_names}')

    @staticmethod
    def _model_predictions_to_domain(decoded_predictions):
        predictions_num = len(decoded_predictions)
        domain_model_outputs = [ModelOutput.from_raw_model_output(decoded_predictions[i])
                                for i in range(predictions_num)]

        return domain_model_outputs

    @staticmethod
    def _load_labels(model_name):
        if model_name == hp.VGG_MODEL_NAME:
            fpath = get_file('rcmalli_vggface_labels_v1.npy', V1_LABELS_PATH, cache_subdir=VGGFACE_DIR)
        elif model_name == hp.RESNET_MODEL_NAME:
            fpath = get_file('rcmalli_vggface_labels_v2.npy', V2_LABELS_PATH, cache_subdir=VGGFACE_DIR)
        else:
            raise ValueError('Unknown model name')

        labels = np.load(fpath)

        return labels
