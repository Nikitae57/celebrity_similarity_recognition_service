import numpy as np
import matplotlib.pyplot as plt

from model import hparams
from model.celeb_similarity.predictor import Predictor
from model.celeb_similarity.image_preprocessor import ImagePreprocessor
from view.celeb_similarity.prediction_view import PredictionView


class Controller:
    def __init__(self):
        self.__predictor = Predictor()
        self.__image_preprocessor = ImagePreprocessor()

    def process_image(self, stream):
        image = np.asarray(plt.imread(stream))
        preprocessed_image = self.__image_preprocessor.preprocess_image(image, target_shape=hparams.img_size)
        predictions = self.__predictor.predict(preprocessed_image)
        prediction_view = PredictionView(predictions)

        return prediction_view.to_json()
