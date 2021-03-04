class PredictedCeleb:
    def __init__(self, name: str, probability: float):
        self.label = name
        self.probability = probability

    @staticmethod
    def from_raw_model_output(model_output):
        label = model_output[0].decode('utf-8')
        probability = model_output[1]

        return PredictedCeleb(name=label, probability=probability)
