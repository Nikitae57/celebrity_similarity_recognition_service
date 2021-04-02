from typing import Optional, List

from model.celeb_similarity.aggregator_output import AggregatorOutput
from view.celeb_similarity.prediction import SimilarityPredictionForFace
from view.errors import Error


class CelebSimilarityDomain:
    def __init__(
        self,
        predictions_for_faces: [SimilarityPredictionForFace] = None,
        user_cropped_face_url: str = None,
        error: Error = None,
        top_predictions_count=3
    ):
        if predictions_for_faces is None or user_cropped_face_url is None:
            self.data = None
        else:
            self.data = CelebSimilarityBody(
                predictions_for_faces=predictions_for_faces,
                user_cropped_face_url=user_cropped_face_url,
                top_predictions_count=top_predictions_count
            )

        self.error = error

    @staticmethod
    def from_aggregator_output(aggregator_output: AggregatorOutput):
        response = CelebSimilarityDomain(
            predictions_for_faces=aggregator_output.predictions,
            user_cropped_face_url=aggregator_output.user_cropped_face_url
        )
        return response

    @staticmethod
    def from_model_outputs(predictions_for_each_face: [SimilarityPredictionForFace]):
        response = CelebSimilarityDomain(predictions_for_faces=predictions_for_each_face)
        return response

    @staticmethod
    def from_error(error: Error):
        response = CelebSimilarityDomain(error=error)
        return response


class CelebSimilarityBody:
    def __init__(
        self,
        predictions_for_faces: [SimilarityPredictionForFace] = None,
        user_cropped_face_url: str = None, top_predictions_count=3
    ):
        self.user_cropped_face_id = user_cropped_face_url
        self.predictions = self._get_top_predictions(predictions_for_faces, top_predictions_count)

    @staticmethod
    def _get_top_predictions(
        predictions: Optional[List[SimilarityPredictionForFace]],
        top_predictions_count=3
    ) -> Optional[List[SimilarityPredictionForFace]]:

        if predictions is None:
            return None

        for model_output in predictions:
            if len(model_output.predicted_celebs) != top_predictions_count:
                model_output.predicted_celebs = model_output.predicted_celebs[:top_predictions_count]

        return predictions
