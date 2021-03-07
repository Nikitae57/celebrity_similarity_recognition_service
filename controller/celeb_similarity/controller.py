import traceback

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from model import hparams
from model.celeb_similarity.similarity_predictor.predictor import Predictor
from model.celeb_similarity.similarity_predictor.image_preprocessor import ImagePreprocessor, MoreThanOneFaceError
from model.celeb_similarity.most_similar_face.most_similar_face_predictor import MostSimilarFacePredictor
from view.celeb_similarity.prediction import SimilarityPredictionForDetectedFace
from view.errors import Error
from view.celeb_similarity.response import CelebSimilarityResponse


class Controller:
    def __init__(self, model_name: str, faces_img_dir: str, faces_vectors_dir: str):
        self._predictor = Predictor(model_name)
        self._image_preprocessor = ImagePreprocessor()
        self._most_similar_face_predictor = MostSimilarFacePredictor(
            faces_img_root_dir=faces_img_dir,
            faces_vectors_root_dir=faces_vectors_dir
        )

    def process_image(self, stream):
        user_image_array = np.asarray(plt.imread(stream))

        try:
            preprocessed_image = self._image_preprocessor.preprocess_image(user_image_array, target_shape=hparams.img_size)
        except MoreThanOneFaceError:
            error = Error.more_than_one_face()
            return CelebSimilarityResponse.from_error(error).to_json()
        except Exception:
            error = Error.unknown_error()
            return CelebSimilarityResponse.from_error(error).to_json()

        # TODO remove
        plt.imshow(preprocessed_image.reshape((224, 224, 3)))
        plt.show()

        try:
            predictions = self._predictor.predict(preprocessed_image)
            user_image = Image.fromarray(user_image_array)
            prediction_views = []
            for prediction in predictions:
                descending_probability_celebs = sorted(prediction.predicted_celebs, key=lambda x: -x.probability)
                top_predicted_celeb_name = descending_probability_celebs[0].name
                most_similar_face_url = self._most_similar_face_predictor.get_most_similar_face_path(
                    user_image,
                    top_predicted_celeb_name
                )

                prediction_view = SimilarityPredictionForDetectedFace(prediction, most_similar_face_url)
                prediction_views.append(prediction_view)
        except Exception as e:
            traceback.print_exc()
            error = Error.unknown_error()
            prediction_response = CelebSimilarityResponse.from_error(error)
            return prediction_response.to_json()

        prediction_response = CelebSimilarityResponse(prediction_views)

        return prediction_response.to_json()
