from view.celeb_similarity.prediction import SimilarityPredictionForFace
from view.errors import Error


class CelebSimilarityResult:
    def __init__(
        self,
        predictions_for_faces: [SimilarityPredictionForFace] = None,
        user_cropped_face_url: str = None,
        error: Error = None
    ):
        self.predictions_for_faces = predictions_for_faces
        self.user_cropped_face_url = user_cropped_face_url
        self.error = error
