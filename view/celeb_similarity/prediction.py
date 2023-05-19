from model.celeb_similarity.model.model_output import ModelOutput


class SimilarityPredictionForFace:
    def __init__(self, model_output: ModelOutput, most_similar_face_url: str):
        self.most_similar_face_url = most_similar_face_url
        self.predicted_celebs = model_output.predicted_celebs

