from openai import OpenAI
client = OpenAI()

error_1 = "file-7HyND1C4EyBGYc5cvUB6VN"
error_100 = "file-DY7WJ92DG6jKC5Kbr2numa"
file_response = client.files.content(error_100)
print(file_response.text)