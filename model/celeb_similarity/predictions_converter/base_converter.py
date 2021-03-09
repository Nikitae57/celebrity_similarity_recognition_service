from view.celeb_similarity.response import CelebSimilarityDomain


class BaseConverter:
    def convert(self, celeb_similarity_domain: CelebSimilarityDomain) -> str:
        pass
