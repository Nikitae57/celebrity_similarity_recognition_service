import numpy as np
from img2vec_pytorch import Img2Vec
from PIL import Image


class ImageVectorizer:
    def __init__(self):
        self._vectorizer = Img2Vec(cuda=False)

    def vectorize(self, image: Image) -> np.ndarray:
        vector = self._vectorizer.get_vec(image)
        return vector
