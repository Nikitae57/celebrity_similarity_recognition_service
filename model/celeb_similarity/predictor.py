import os
import traceback
import uuid

from PIL import Image

from model import hparams
from model.most_similar_face.most_similar_face_predictor import MostSimilarFacePredictor
from model.celeb_similarity.image_preprocessor import ImagePreprocessor, MoreThanOneFaceError, NoFacesError
from model.celeb_similarity.inference import Inference
from view.celeb_similarity.prediction import SimilarityPredictionForFace
from view.celeb_similarity.response import CelebSimilarityResult
from view.errors import Error


class Predictor:
    def __init__(self,
                 model_name: str,
                 faces_img_dir: str,
                 faces_vectors_dir: str,
                 user_face_cache_path: str):
        self._predictor = Inference(model_name)
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
            return CelebSimilarityResult(error=error)
        except NoFacesError:
            error = Error.no_faces()
            return CelebSimilarityResult(error=error)
        except Exception:
            error = Error.unknown_error()
            return CelebSimilarityResult(error=error)

        # plt.imshow(preprocessed_image.reshape((224, 224, 3)))
        # plt.show()

        try:
            predictions = self._predictor.run(preprocessed_image)
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
            error_result = CelebSimilarityResult(error=error)

            return error_result

        prediction_result = CelebSimilarityResult(
            predictions_for_faces=prediction_views,
            user_cropped_face_url=user_face_id
        )

        return prediction_result

    def _save_user_face(self, face_array, img_format='JPEG'):
        img_id = uuid.uuid4().hex[:20]
        user_img_name = f'{img_id}.jpg'
        user_face_img_path = os.path.join(self._user_faces_cache_path, user_img_name)
        image = Image.fromarray(face_array)
        image.save(user_face_img_path, img_format)

        return img_id
