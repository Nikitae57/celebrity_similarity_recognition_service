import os
import traceback
import uuid

from PIL import Image

from model import hparams
from model.celeb_similarity.aggregator_output import AggregatorOutput
from model.celeb_similarity.most_similar_face.most_similar_face_predictor import MostSimilarFacePredictor
from model.celeb_similarity.similarity_predictor.image_preprocessor import ImagePreprocessor, MoreThanOneFaceError
from model.celeb_similarity.similarity_predictor.predictor import Predictor
from view.celeb_similarity.prediction import SimilarityPredictionForFace
from view.celeb_similarity.response import CelebSimilarityDomain
from view.errors import Error


class PredictionsAggregator:
    def __init__(self,
                 model_name: str,
                 faces_img_dir: str,
                 faces_vectors_dir: str,
                 user_face_cache_path: str):
        self._predictor = Predictor(model_name)
        self._image_preprocessor = ImagePreprocessor()
        self._most_similar_face_predictor = MostSimilarFacePredictor(
            faces_img_root_dir=faces_img_dir,
            faces_vectors_root_dir=faces_vectors_dir
        )
        self._user_faces_cache_path = user_face_cache_path
        os.makedirs(user_face_cache_path, exist_ok=True)

    def predict(self, user_image_array):
        try:
            preprocessed_image, user_face_array = self._image_preprocessor.preprocess_image(
                user_image_array,
                target_shape=hparams.img_size
            )
            user_face_id = self._save_user_face(user_face_array)
        except MoreThanOneFaceError:
            error = Error.more_than_one_face()
            return CelebSimilarityDomain.from_error(error)
        except Exception:
            error = Error.unknown_error()
            return CelebSimilarityDomain.from_error(error)

        # plt.imshow(preprocessed_image.reshape((224, 224, 3)))
        # plt.show()

        try:
            predictions = self._predictor.predict(preprocessed_image)
            user_image = Image.fromarray(user_face_array)
            prediction_views = []
            for prediction in predictions:
                descending_probability_celebs = sorted(prediction.predicted_celebs, key=lambda x: -x.probability)
                top_predicted_celeb_name = descending_probability_celebs[0].name
                most_similar_face_url = self._most_similar_face_predictor.get_most_similar_face_path(
                    user_image,
                    top_predicted_celeb_name
                )

                prediction_view = SimilarityPredictionForFace(prediction, most_similar_face_url)
                prediction_views.append(prediction_view)
        except Exception as e:
            traceback.print_exc()
            error = Error.unknown_error()
            error_response = CelebSimilarityDomain.from_error(error)

            return error_response

        aggregator_output = AggregatorOutput(prediction_views, user_face_id)
        prediction_response = CelebSimilarityDomain.from_aggregator_output(aggregator_output)

        return prediction_response

    def _save_user_face(self, face_array, img_format='JPEG'):
        img_id = uuid.uuid4().hex[:20]
        user_img_name = f'{img_id}.jpg'
        user_face_img_path = os.path.join(self._user_faces_cache_path, user_img_name)
        image = Image.fromarray(face_array)
        image.save(user_face_img_path, img_format)

        return img_id
