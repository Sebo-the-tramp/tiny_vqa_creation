import os 
import json

models = {
    "Phi-3-vision-128k-instruct": "general",
    "Phi-3.5V": "general",
    "mPLUG-Owl3-1B-241014": "general",
    "mPLUG-Owl3-2B-241014": "general",
    "mPLUG-Owl3-7B-241101": "general",
    "llava-interleave-qwen-7b-hf": "general",
    "llava-interleave-qwen-7b-dpo-hf": "general",
    "vila-1.5-3b": "general",
    "vila-1.5-3b-s2": "general",
    "vila-1.5-8b": "general",
    "vila-1.5-13b": "general",
    "LLaVA-NeXT-Video-7B-DPO-hf": "general",
    "LLaVA-NeXT-Video-7B-hf": "general",
    "InternVL2-1B": "general",
    "InternVL2-2B": "general",
    "InternVL2-4B": "general",
    "InternVL2-8B": "general",
    "InternVL2-26B": "general",
    "InternVL2-40B": "general",
    "InternVL2-76B": "general",
    "InternVL2_5-1B": "general",
    "InternVL2_5-2B": "general",
    "InternVL2_5-4B": "general",
    "InternVL2_5-8B": "general",
    "InternVL2_5-26B": "general",
    "InternVL2_5-38B": "general",
    "InternVL2_5-78B": "general",
    "Mantis-8B-Idefics2": "general",
    "Mantis-llava-7b": "general",
    "Mantis-8B-siglip-llama3": "general",
    "Mantis-8B-clip-llama3": "general",
    "gpt4v": "general",
    "gpt4o": "general",
    "o1": "general",
    "gpt4o-mini": "general",
    "gemini-1.5-flash": "general",
    "gemini-1.5-pro": "general",
    "claude-3-5-sonnet": "general",
    "claude-3-sonnet": "general",
    "claude-3-opus": "general",
    "claude-3-haiku": "general",
    "video-llava-7b": "general",
    "chat-univi-7b": "general",
    "chat-univi-13b": "general",
    "pllava-7b": "general",
    "pllava-13b": "general",
}

with open("/Users/sebastiancavada/Desktop/tmp_Paris/vqa_analysis/utils/metadata.json", 'r') as f:
    config = json.load(f)
    print(config)

for id in models.keys():
    # print(id)
    for entry in config:
        if entry['id'] == id and entry["release_type"] == "open_weights":
            # print(str(round(entry['params_b'], 2)).replace('.', ',')) # printing number of params
            print(entry['family'])