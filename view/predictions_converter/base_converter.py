from view.celeb_similarity.response import CelebSimilarityResult


class BaseConverter:
    def convert(self, celeb_similarity_domain: CelebSimilarityResult) -> str:
        pass
