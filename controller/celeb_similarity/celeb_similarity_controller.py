import matplotlib.pyplot as plt
import numpy as np

from model.celeb_similarity.predictor import Predictor
from view.predictions_converter.base_converter import BaseConverter


class CelebSimilarityController:
    def __init__(
            self,
            model_name: str,
            faces_img_dir: str,
            faces_vectors_dir: str,
            user_face_cache_dir: str,
            response_converter: BaseConverter
    ):
        self._celeb_similarity_predictor = Predictor(
            model_name=model_name,
            faces_img_dir=faces_img_dir,
            faces_vectors_dir=faces_vectors_dir,
            user_face_cache_path=user_face_cache_dir
        )
        self._response_converter = response_converter

    def process_image(self, stream):
        # start_time = time()
        user_image_array = self._read_image(stream)
        predictions = self._celeb_similarity_predictor.predict(user_image_array)

        response = self._response_converter.convert(predictions)
        # print((time() - start_time) * 1000)

        return response

    @staticmethod
    def _read_image(stream):
        return np.asarray(plt.imread(stream))
