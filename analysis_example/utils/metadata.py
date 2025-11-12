import os, json

metadata = {}


def read_metadata():
    global metadata
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "metadata.json"), "r") as f:
        metadata = json.load(f)

def merge_metadata(answers_vlm):
    """
    Merge metadata information into the answers_vlm list of dictionaries.

    Parameters:
    - answers_vlm (list): List of dictionaries containing model answers.
    - metadata (list): List of dictionaries containing model metadata.

    Returns:
    - list: Updated answers_vlm with merged metadata.
    """

    # Force metadata re-read
    read_metadata()

    metadata_dict = {item["id"]: item for item in metadata}

    for model_data in answers_vlm:
        model_name = model_data.get("model")
        if model_name in metadata_dict:
            # data to add:
            data_to_add = metadata_dict[model_name].copy()
            if "id" in data_to_add:
                del data_to_add["id"]  # Remove 'id' to avoid duplication
            if "updated_at" in data_to_add:
                del data_to_add["updated_at"]  # Remove 'updated_at' if not needed            
            model_data.update(data_to_add)
        else:
            print(f"Warning: No metadata found for model '{model_name}'")

    return answers_vlm

read_metadata()