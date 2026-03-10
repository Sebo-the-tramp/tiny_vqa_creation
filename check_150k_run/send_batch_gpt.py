from openai import OpenAI
client = OpenAI()

batch_input_file = client.files.create(
    file=open("./batches/output.jsonl", "rb"),
    purpose="batch"
)

print(batch_input_file)

batch_input_file_id = batch_input_file.id
# batch_input_file_id = "file-FETi7dnLbUQyQ6uYve7aZk"
client.batches.create(
    input_file_id=batch_input_file_id,
    endpoint="/v1/chat/completions",
    completion_window="24h",
    metadata={
        "description": "testing on 100 samples from the 150k dataset",
        "model": "gpt-5-mini-2025-08-07",
    }
)