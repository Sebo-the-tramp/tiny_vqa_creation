class ImpossibleToAnswer(Exception):
    """Custom exception for specific error conditions."""

    def __init__(self, message="This question cannot be answered with the given data."):
        # print("\033[95m  ATTENTION: question impossible to answer\033[0m")
        # print("============================================================")
        super().__init__(message)
