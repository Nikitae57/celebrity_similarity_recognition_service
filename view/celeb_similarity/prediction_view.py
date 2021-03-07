import jsonpickle
from model.celeb_similarity.similarity_predictor.model_output import ModelOutput
from view.celeb_similarity.errors import Error
from view.celeb_similarity import errors


class CelebSimilarityResponse:
    def __init__(self, model_outputs: [ModelOutput] = None, error: Error = None):
        self.model_outputs = model_outputs
        self.error = error

    def to_json(self, top_results_count=3):
        if self.model_outputs is not None:
            for model_output in self.model_outputs:
                if len(model_output.predicted_celebs) != top_results_count:
                    model_output.predicted_celebs = model_output.predicted_celebs[:top_results_count]

        outputs_json = jsonpickle.encode(self, unpicklable=False)

        return outputs_json

    @staticmethod
    def from_model_outputs(model_outputs: [ModelOutput]):
        response = CelebSimilarityResponse(model_outputs=model_outputs)
        return response

    @staticmethod
    def from_error(error: Error):
        response = CelebSimilarityResponse(error=error)
        return response
