from view.celeb_similarity.prediction import SimilarityPredictionForFace


class AggregatorOutput:
    def __init__(self,
                 predictions: [SimilarityPredictionForFace],
                 user_face_url: str):

        self.predictions = predictions
        self.user_cropped_face_url = user_face_url
