import numpy as np
import tensorflow as tf

from mtcnn.mtcnn import MTCNN
from PIL import Image
from keras_vggface.utils import preprocess_input


class MoreThanOneFaceError(Exception):
    pass


class ImagePreprocessor:
    def __init__(self):
        self.__graph = tf.get_default_graph()
        self.__face_detector = self._load_model()

    def _load_model(self):
        with self.__graph.as_default():
            model = MTCNN()
        return model

    def _get_faces_from_image(self, pixels: np.ndarray, target_shape=(224, 224)):
        face_arrays = []
        with self.__graph.as_default():
            results = self.__face_detector.detect_faces(pixels)
            for result in results:
                x1, y1, width, height = result['box']
                x2, y2 = x1 + width, y1 + height
                face = pixels[y1:y2, x1:x2]
                img = Image.fromarray(face)
                img = img.resize(target_shape)
                face_array = np.asarray(img).astype(np.float32)
                face_arrays.append(face_array)

        return face_arrays

    def preprocess_image(self, pixels, target_shape=(224, 224)):
        # pixels = np.asarray(Image.open(pixels))
        face_arrays = self._get_faces_from_image(pixels, target_shape)
        if len(face_arrays) != 1:
            raise MoreThanOneFaceError()

        face_array = face_arrays[0]
        samples = np.expand_dims(face_array, axis=0)
        # prepare the face for the model, e.g. center pixels
        samples = preprocess_input(samples, version=2)

        return samples
