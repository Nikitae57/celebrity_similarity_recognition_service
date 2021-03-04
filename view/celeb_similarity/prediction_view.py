import jsonpickle
from model.celeb_similarity.model_output import ModelOutput


class PredictionView:
    def __init__(self, model_outputs: [ModelOutput]):
        self.__model_outputs = model_outputs

    def to_json(self, top_results_count=3):
        model_outputs_to_return = self.__model_outputs[:top_results_count]
        outputs_json = jsonpickle.encode(model_outputs_to_return, unpicklable=False)

        return outputs_json
