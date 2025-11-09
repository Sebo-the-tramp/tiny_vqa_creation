from huggingface_hub import HfApi

from metadata import metadata

# Initialize API
api = HfApi()


def get_metadata(model):

    model_id = model['source']
    # Get model info
    info = api.model_info(model_id)

    print(f"Model ID: {info.modelId}")
    print(f"Author: {info.author}")
    print(f"Downloads: {info.downloads}")
    print(f"Likes: {info.likes}")
    print(f"Last Modified: {info.lastModified}")
    print(f"Created at: {info.created_at}")
    model['release_year'] = info.created_at.year
    print(f"Model Size in Billions: {info.safetensors.total/1e9:.2f}")
    model['params_b'] = info.safetensors.total/1e9
    print(f"License: {info.card_data.get('license', 'N/A')}")
    model['license'] = info.card_data.get('license', 'N/A')
    print(f"Tags: {info.tags}")
    model['tags'] = info.tags

for model in metadata:
    try:
        get_metadata(model)
        print("-" * 10 + model['source'] + "-" * 10)
        print()
        print(model)
    except Exception as e:
        print(f"Error fetching metadata for {model['source']}: {e}")

print(metadata)
with open("utils/metadata.json", "w") as f:
    import json
    json.dump(metadata, f, indent=4)
