class ImpossibleToAnswer(Exception):
    """Custom exception for specific error conditions."""

    def __init__(self, message="This question cannot be answered with the given data."):
        super().__init__(message)
