import os
import numpy as np
from model.most_similar_face.image_vectorizer import ImageVectorizer
from model import util
from PIL import Image

from model.most_similar_face.model.vectorized_image import VectorizedImage
from model.most_similar_face.model.vectorized_person import VectorizedPerson


class MostSimilarFacePredictor:
    def __init__(self, faces_img_root_dir: str, faces_vectors_root_dir: str):
        self._vectorizer = ImageVectorizer()
        self._person_name_to_vector = self._load_vectorized_faces(faces_img_root_dir, faces_vectors_root_dir)

    def get_most_similar_face_path(self, face_img: Image, person_name: str) -> str:
        face_vector = self._vectorizer.vectorize(face_img)
        vectorized_person = self._person_name_to_vector[person_name]
        _, vector_index = vectorized_person.tree.query(face_vector)
        most_similar_face_img_path = vectorized_person.vectorized_faces[vector_index].img_path

        return most_similar_face_img_path

    @staticmethod
    def _load_vectorized_faces(faces_img_root_dir: str, faces_vectors_root_dir: str):
        util.assert_dirs_exist([faces_img_root_dir, faces_vectors_root_dir])
        util.assert_dirs_contain_same_items_count([faces_img_root_dir, faces_vectors_root_dir])

        person_name_to_vector = dict()
        img_dirs = os.listdir(faces_img_root_dir)
        img_dirs = list(filter(lambda x: os.path.isdir(os.path.join(faces_img_root_dir, x)), img_dirs))
        for img_dir in img_dirs:
            person_name = os.path.basename(os.path.normpath(img_dir)).strip()

            vector_dir = os.path.join(faces_vectors_root_dir, img_dir)
            img_dir = os.path.join(faces_img_root_dir, img_dir)

            if not os.path.exists(vector_dir):
                raise ValueError(f'Failed to find vector equivalent dir for {img_dir}')

            vectorized_images = []
            for img_file_name in os.listdir(img_dir):
                vector_file = os.path.join(vector_dir, img_file_name)
                vector_file = f'{vector_file}.{util.NUMPY_FILE_EXTENSION}'

                vector = np.load(vector_file)
                img_path = os.path.join(img_dir, img_file_name)
                vectorized_image = VectorizedImage(img_path=img_path, img_vector=vector)
                vectorized_images.append(vectorized_image)

            vectorized_person = VectorizedPerson(person_name, vectorized_images)
            person_name_to_vector[vectorized_person.name] = vectorized_person

        return person_name_to_vector


