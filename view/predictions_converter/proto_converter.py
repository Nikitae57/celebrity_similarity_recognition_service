from view.predictions_converter.base_converter import BaseConverter
from view.celeb_similarity.response import CelebSimilarityResult
from view.errors import Error, ErrorType
from view.proto import protocol_pb2 as pb


class ProtoConverter(BaseConverter):
    def convert(self, celeb_similarity_domain: CelebSimilarityResult) -> str:
        error_pb = self._error_to_pb(celeb_similarity_domain.error)

        predictions_pb_array = []
        for i in range(len(celeb_similarity_domain.predictions)):
            prediction = celeb_similarity_domain.predictions[i]

            predictions_pb = pb.CelebSimilarityPredictions(
                most_similar_face_url=prediction.most_similar_face_url,
                predicted_celebs=[
                    pb.PredictedCeleb(name=celeb.name, probability=celeb.probability)
                    for celeb in prediction.predicted_celebs
                ]
            )
            predictions_pb_array.append(predictions_pb)

        feature_pb = pb.CelebSimilarityFeature(celeb_similarity_predictions=predictions_pb_array)
        response_bp = pb.GetFaceFeaturesResponse(
            error=error_pb,
            user_cropped_face_id=celeb_similarity_domain.user_cropped_face_id,
            celeb_similarity_features=feature_pb
        )

        return response_bp.SerializeToString()

    @staticmethod
    def _error_to_pb(error: Error):
        if error.type == ErrorType.UNKNOWN_ERROR:
            error_type_pb = pb.UNKNOWN_ERROR
        elif error.type == ErrorType.MORE_THAN_ONE_FACE:
            error_type_pb = pb.MORE_THAN_ONE_FACE
        else:
            raise ValueError('Unknown ErrorType enum value')

        return pb.Error(
            type=error_type_pb,
            error_message=error.message
        )
