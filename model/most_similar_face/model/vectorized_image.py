import numpy as np


class VectorizedImage:
    def __init__(self, img_path: str, img_vector: np.ndarray):
        self.img_vector = img_vector
        self.img_path = img_path
