import numpy as np
import tensorflow as tf

from mtcnn.mtcnn import MTCNN
from PIL import Image
from keras_vggface.utils import preprocess_input


class ImagePreprocessor:
    def __init__(self):
        self.__session = tf.Session()
        self.__graph = tf.get_default_graph()
        self.__face_detector = self.__load_model()

    def __load_model(self):
        with self.__session.as_default():
            with self.__graph.as_default():
                model = MTCNN()
        return model

    def __get_face_from_image(self, pixels: np.ndarray, target_shape=(224, 224)):
        with self.__session.as_default():
            with self.__graph.as_default():
                results = self.__face_detector.detect_faces(pixels)
                x1, y1, width, height = results[0]['box']
                x2, y2 = x1 + width, y1 + height
                face = pixels[y1:y2, x1:x2]
                img = Image.fromarray(face)
                img = img.resize(target_shape)
                face_array = np.asarray(img).astype(np.float32)

        return face_array

    def preprocess_image(self, pixels, target_shape=(224, 224)):
        # pixels = np.asarray(Image.open(pixels))
        face_array = self.__get_face_from_image(pixels, target_shape)
        samples = np.expand_dims(face_array, axis=0)
        # prepare the face for the model, e.g. center pixels
        samples = preprocess_input(samples, version=2)

        return samples
