IMPOSSIBLE_REASON_CODES = {
    "0": "Uncategorized",
    "1": "Visibility",
    "2": "Objects",
    "3": "Timestep",
    "4": "Ambiguity",
    "5": "Augmentation",
    "6": "Uncertainty",
    "7": "No Change Counterfactual",
    "8": "Options",
    "9": "Other",
}


def parse_impossible_reason(message: str):
    msg = message or ""
    code = msg[0] if msg[:1].isdigit() else "0"
    label = IMPOSSIBLE_REASON_CODES.get(code, "Unknown")
    return code, label, msg


class ImpossibleToAnswer(Exception):
    """Custom exception for specific error conditions."""

    def __init__(self, message="This question cannot be answered with the given data."):
        # print("\033[95m  ATTENTION: question impossible to answer\033[0m")
        # print("============================================================")
        super().__init__(message)
