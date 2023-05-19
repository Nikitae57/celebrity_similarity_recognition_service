import jsonpickle
from flask import Response

from view.predictions_converter.base_converter import BaseConverter
from view.celeb_similarity.response import CelebSimilarityResult


class JsonConverter(BaseConverter):
    def convert(self, celeb_similarity_domain: CelebSimilarityResult) -> str:
        json = jsonpickle.encode(celeb_similarity_domain, unpicklable=False, make_refs=False)
        response = Response(json, mimetype='application/json')
        print(response.headers)

        return response
