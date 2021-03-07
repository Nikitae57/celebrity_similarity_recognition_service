import numpy as np
import matplotlib.pyplot as plt

from model import hparams
from model.celeb_similarity.similarity_predictor.predictor import Predictor
from model.celeb_similarity.similarity_predictor.image_preprocessor import ImagePreprocessor, MoreThanOneFaceError
from model.celeb_similarity.most_similar_face.most_similar_face_predictor import MostSimilarFacePredictor
from view.celeb_similarity.errors import Error
from view.celeb_similarity.prediction_view import CelebSimilarityResponse


class Controller:
    def __init__(self, model_name: str, faces_img_dir: str = None, faces_vectors_dir: str = None):
        self._predictor = Predictor(model_name)
        self._image_preprocessor = ImagePreprocessor()
        # self._most_similar_face_predictor = MostSimilarFacePredictor(
        #     faces_img_root_dir=faces_img_dir,
        #     faces_vectors_root_dir=faces_vectors_dir
        # )

    def process_image(self, stream):
        image = np.asarray(plt.imread(stream))

        try:
            preprocessed_image = self._image_preprocessor.preprocess_image(image, target_shape=hparams.img_size)
        except MoreThanOneFaceError:
            error = Error.more_than_one_face()
            return CelebSimilarityResponse.from_error(error).to_json()
        except Exception:
            error = Error.unknown_error()
            return CelebSimilarityResponse.from_error(error).to_json()

        # TODO remove
        plt.imshow(preprocessed_image.reshape((224, 224, 3)))
        plt.show()

        predictions = self._predictor.predict(preprocessed_image)
        # for i in predictions.
        # most_similar_face_url = self._most_similar_face_predictor.get_most_similar_face_path(image, )

        prediction_view = CelebSimilarityResponse(predictions)

        return prediction_view.to_json()
