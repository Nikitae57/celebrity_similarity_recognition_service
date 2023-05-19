from scipy.spatial import KDTree

from model.most_similar_face.model.vectorized_image import VectorizedImage


class VectorizedPerson:
    def __init__(self, person_name: str, vectorized_faces: [VectorizedImage]):
        vectors = [vectorized_face.img_vector for vectorized_face in vectorized_faces]
        self.tree = KDTree(vectors)
        self.vectorized_faces = vectorized_faces
        self.name = person_name
