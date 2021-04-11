"""
cleanse with respect to each corpus.
"""


class Cleanser:
    """
    Cleanser, applies to all domain.
    """
    def cleanse(self, token: str) -> str:
        raise NotImplementedError


class CocaSpokCleanser(Cleanser):

    def cleanse(self, token: str) -> str:
        pass


class CocaFictCleanser(Cleanser):

    def cleanse(self, token: str) -> str:
        pass
