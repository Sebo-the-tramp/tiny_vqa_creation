# average of ~5 images per 10 questions
import re
import json 

JSON_PATH="/data0/sebastian.cavada/compositional-physics/tiny_vqa_creation/output/run_28_general/test_run_28_general.json"
ONLINE_DB_URL="https://huggingface.co/datasets/sebothetramp/test_newtphys/resolve/main/"

with open(JSON_PATH, "r") as f:
    full_dataset = json.load(f)

batches = []
tmp_batched = []

count_now = 0
total_images = 0

# {"custom_id": "request-1", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-3.5-turbo-0125", "messages": [{"role": "system", "content": "You are a helpful assistant."},{"role": "user", "content": "Hello world!"}],"max_tokens": 1000}}

def build_gemini_parts(prompt_string, image_urls):
    """
    Parses a string with <image> tokens and weaves them into a Gemini 'parts' array.
    """
    parts = []
    
    # Split the string by the token. 
    # E.g., "<image> Hello" becomes ["", " Hello"]
    text_chunks = prompt_string.split("<image>")
    
    # Ensure we have the right number of URLs for the tokens
    if len(text_chunks) - 1 != len(image_urls):
        raise ValueError(f"Found {len(text_chunks) - 1} <image> tokens, but got {len(image_urls)} URLs.")

    for i, chunk in enumerate(text_chunks):
        # 1. Add the text chunk if it's not empty
        if chunk:
            parts.append({"text": chunk})
            
        # 2. Add the corresponding image (if we aren't at the end of the list)
        if i < len(image_urls):
            # Fetch and convert the image from Hugging Face
            response = requests.get(image_urls[i])
            response.raise_for_status()
            image_b64 = base64.b64encode(response.content).decode('utf-8')
            
            parts.append({
                "inlineData": {
                    "mimeType": "image/jpeg", # Adjust if using PNGs
                    "data": image_b64
                }
            })
            
    return parts


def build_openai_request(idx, question, images):

    img_idx = 0
    result = []

    image_parts = question.split('<image>')
    for j, text in enumerate(image_parts):
        if j > 0:  # if this is not the first sub-part, it means we had an <image> placeholder
            if img_idx < len(images):
                image_url = {
                    "type": "image_url",
                    "image_url": {
                        "url": images[img_idx],
                        "size": (1000,562),
                        "detail": "low"}
                    }
                result.append(image_url)
                img_idx += 1

        if text:  # Add the text part
            result.append({"type": "text", "text": text})

    processed_prompt = result
    print(processed_prompt)

    base_request = {
        "custom_id": idx,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {"model": "gpt-5-mini-2025-08-07", #gpt-3.5-turbo-0125

        "messages":[
            {
                "role": "user",
                "content": processed_prompt
            }
        ],
        "max_completion_tokens": 100}
    }
    return base_request


def save_json(batched, filename="./batches/output.jsonl"):
    """Saves a list of data to a JSONL file, one JSON object per line."""
    # Using 'w' to write/overwrite. Use 'a' if you want to append multiple batches over time.
    with open(filename, 'w', encoding='utf-8') as f:
        for item in batched:
            # json.dumps converts the dict to a string, \n adds the line break
            f.write(json.dumps(item) + '\n')
            
    print(f"Successfully saved {len(batched)} items to {filename}")


for i, question in enumerate(full_dataset):

    if "random-cam-stationary" in question['simulation_id']:
        # not handles rn
        continue

    question_mod = question.copy()

    question_mod['question'] = question_mod['question'] + "\n Answer with the option's letter from the given choices directly. You can only answer one letter from A, B, C, D.\n"
    print(question_mod['question'])
    print("++++=====++++")

    new_file_names = []
    total_images += len(question_mod['file_name'])
    for file in question_mod['file_name']:

        #THIS IS JUST FOR NOW
        replace_str = "/scratch/project/eu-25-92/composite_physics/dataset/simulation_v4/dl3dv/random/1/"
        new_file = file.replace(replace_str, ONLINE_DB_URL)
        new_file = re.sub(r'(?<!:)//+', '/', new_file)
        print(file)
        new_file_names.append(new_file)

    question_mod['file_name'] = new_file_names

    openai_request = build_openai_request(question_mod['idx'], question_mod['question'], question_mod['file_name']) 

    tmp_batched.append(openai_request)
    count_now += 1

    if count_now >= 100:
        break
        
    if i%10000 == 0:
        save_json(tmp_batched)

save_json(tmp_batched)
print(f"total of {total_images} total images LOL avg of {total_images/10}")
# # add the remaining batch
# batches.append(tmp_batched)
