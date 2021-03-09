from time import time

import matplotlib.pyplot as plt
import numpy as np

from model.celeb_similarity.predictions_aggregator import PredictionsAggregator


class Controller:
    def __init__(self, model_name: str, faces_img_dir: str, faces_vectors_dir: str):
        self._celeb_similarity_predictions_aggregator = PredictionsAggregator(
            model_name=model_name,
            faces_img_dir=faces_img_dir,
            faces_vectors_dir=faces_vectors_dir
        )

    def process_image(self, stream):
        start_time = time()
        user_image_array = np.asarray(plt.imread(stream))
        response = self._celeb_similarity_predictions_aggregator.predict(user_image_array)
        print((time() - start_time) * 1000)

        return response
