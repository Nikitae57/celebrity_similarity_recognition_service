import jsonpickle
from model.celeb_similarity.similarity_predictor.model_output import ModelOutput
from view.celeb_similarity.prediction import SimilarityPredictionForDetectedFace
from view.errors import Error


class CelebSimilarityResponse:
    def __init__(
        self,
        predictions_for_each_face: [SimilarityPredictionForDetectedFace] = None,
        error: Error = None,
    ):
        self.model_outputs = predictions_for_each_face
        self.error = error

    def to_json(self, top_results_count=3):
        if self.model_outputs is not None:
            for model_output in self.model_outputs:
                if len(model_output.predicted_celebs) != top_results_count:
                    model_output.predicted_celebs = model_output.predicted_celebs[:top_results_count]

        outputs_json = jsonpickle.encode(self, unpicklable=False)

        return outputs_json

    @staticmethod
    def from_model_outputs(predictions_for_each_face: [SimilarityPredictionForDetectedFace]):
        response = CelebSimilarityResponse(predictions_for_each_face=predictions_for_each_face)
        return response

    @staticmethod
    def from_error(error: Error):
        response = CelebSimilarityResponse(error=error)
        return response
