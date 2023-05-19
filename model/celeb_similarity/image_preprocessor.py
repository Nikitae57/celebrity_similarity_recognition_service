import numpy as np
import tensorflow as tf
from PIL import Image
from keras_vggface.utils import preprocess_input
from mtcnn.mtcnn import MTCNN


class MoreThanOneFaceError(Exception):
    pass


class NoFacesError(Exception):
    pass


class ImagePreprocessor:
    def __init__(self):
        self.__graph = tf.get_default_graph()
        self.__face_detector = self._load_model()

    def preprocess_image(self, pixels, target_shape=(224, 224)):
        face_arrays = self._get_faces_from_image(pixels, target_shape)
        num_faces = len(face_arrays)
        if num_faces > 1:
            raise MoreThanOneFaceError()
        elif num_faces < 1:
            raise NoFacesError()

        face_array = face_arrays[0]
        samples = np.asarray(np.expand_dims(face_array, axis=0), dtype=np.float64)
        # prepare the face for the model, e.g. center pixels
        samples = preprocess_input(samples, version=2)

        return samples, face_array

    def _load_model(self):
        with self.__graph.as_default():
            model = MTCNN()
        return model

    def _get_faces_from_image(self, pixels: np.ndarray, target_shape=(224, 224)):
        face_arrays = []
        with self.__graph.as_default():
            results = self.__face_detector.detect_faces(pixels)
            for result in results:
                coords = self._face_detection_result_to_coords(result)
                face_array = self._cut_face_from_img(pixels, coords, target_shape)
                face_arrays.append(face_array)

        return face_arrays

    @staticmethod
    def _face_detection_result_to_coords(result):
        x1, y1, width, height = result['box']
        x2, y2 = x1 + width, y1 + height
        coords = (y1, y2, x1, x2)

        return coords

    @staticmethod
    def _cut_face_from_img(img_array: np.ndarray, face_coords, target_size, padding_fraction=0.2) -> np.ndarray:
        image = Image.fromarray(img_array)

        y1, y2, x1, x2 = face_coords
        face_width = x2 - x1
        face_height = y2 - y1
        face_center_x = x1 + face_width / 2
        face_center_y = y1 + face_height / 2

        aspect = face_width / float(face_height)
        ideal_aspect = target_size[1] / float(target_size[0])

        if aspect > ideal_aspect:
            # Then crop the left and right edges:
            face_width = int(ideal_aspect * face_height)
        else:
            # ... crop the top and bottom:
            face_height = int(face_width / ideal_aspect)

        left = (face_center_x - face_width / 2) - padding_fraction * face_width
        top = face_center_y - face_height / 2 - padding_fraction * face_height
        right = face_center_x + face_width / 2 + padding_fraction * face_width
        bottom = face_center_y + face_height / 2 + padding_fraction * face_height

        thumb = image.crop((left, top, right, bottom)).resize(target_size)

        return np.asarray(thumb)
