from model.celeb_similarity.similarity_predictor.model_output import ModelOutput


class SimilarityPredictionForDetectedFace:
    def __init__(self, model_output: ModelOutput, most_similar_face_url: str):
        self.most_similar_face_url = most_similar_face_url
        self.predicted_celebs = model_output.predicted_celebs

