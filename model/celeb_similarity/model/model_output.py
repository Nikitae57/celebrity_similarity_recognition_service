from model.celeb_similarity.model.predicted_celeb import PredictedCeleb


class ModelOutput:
    def __init__(self, predicted_celebs: [PredictedCeleb]):
        self.predicted_celebs = predicted_celebs

    @staticmethod
    def from_raw_model_output(model_output):
        predicted_celebs = []

        for predicted_celeb_index in range(len(model_output)):
            celeb_name = model_output[predicted_celeb_index][0].replace('b\'', '').replace('\'', '').replace('_', ' ')
            probability = float(model_output[predicted_celeb_index][1])
            recognized_celeb = PredictedCeleb(name=celeb_name, probability=probability)
            predicted_celebs.append(recognized_celeb)

        return ModelOutput(predicted_celebs)
