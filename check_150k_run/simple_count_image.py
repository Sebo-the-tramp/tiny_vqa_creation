from google import genai

client = genai.Client(api_key="AIzaSyDnbN48kOk-98En6tVh8xqzoXEqy0A-5SM")
prompt = ""
your_image_file = client.files.upload(file="/data0/sebastian.cavada/datasets/simulations_v4/dl3dv/random/5/c-1_no-5_d-10_s-dl3dv-all_models-hf-gso_MLP-10_smooth_h-10-40_seed-0_20251212_142910/render/000005.png")

print(
    client.models.count_tokens(
        model="gemini-3.1-pro-preview", contents=[prompt, your_image_file]
    )
)

# response = client.models.generate_content(
#     model="gemini-3-flash-preview", contents=[prompt, your_image_file]
# )
# print(response.usage_metadata)
