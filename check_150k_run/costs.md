## GPT-5.2

The best model for coding and agentic tasks across industries
Price
Input:
$1.750 / 1M tokens
Cached input:
$0.175 / 1M tokens
Output:
$14.000 / 1M tokens


## Gemini3-Pro
Free Tier 	Paid Tier, per 1M tokens in USD
Input price 	Not available 	$2.00, prompts <= 200k tokens
$4.00, prompts > 200k tokens
Output price (including thinking tokens) 	Not available 	$12.00, prompts <= 200k tokens
$18.00, prompts > 200k
Context caching price 	Not available 	$0.20, prompts <= 200k tokens
$0.40, prompts > 200k
$4.50 / 1,000,000 tokens per hour (storage price)

Count tokens

All input to and output from the Gemini API is tokenized, including text, image files, and other non-text modalities.

You can count tokens in the following ways:

    Call count_tokens with the input of the request.
    This returns the total number of tokens in the input only. You can make this call before sending the input to the model to check the size of your requests.

    Use the usage_metadata attribute on the response object after calling generate_content.
    This returns the total number of tokens in both the input and the output: total_token_count.
    It also returns the token counts of the input and output separately: prompt_token_count (input tokens) and candidates_token_count (output tokens).

    If you are using a thinking model, the token used during the thinking process are returned in thoughts_token_count. And if you are using Context caching, the cached token count will be in cached_content_token_count.

Count text tokens

If you call count_tokens with a text-only input, it returns the token count of the text in the input only (total_tokens). You can make this call before calling generate_content to check the size of your requests.

Another option is calling generate_content and then using the usage_metadata attribute on the response object to get the following:

    The separate token counts of the input (prompt_token_count), the cached content (cached_content_token_count) and the output (candidates_token_count)
    The token count for the thinking process (thoughts_token_count)

    The total number of tokens in both the input and the output (total_token_count)

Python
JavaScript
Go

from google import genai

client = genai.Client()
prompt = "The quick brown fox jumps over the lazy dog."

total_tokens = client.models.count_tokens(
    model="gemini-3-flash-preview", contents=prompt
)
print("total_tokens: ", total_tokens)

response = client.models.generate_content(
    model="gemini-3-flash-preview", contents=prompt
)

print(response.usage_metadata)

Count multi-turn (chat) tokens

If you call count_tokens with the chat history, it returns the total token count of the text from each role in the chat (total_tokens).

Another option is calling send_message and then using the usage_metadata attribute on the response object to get the following:

    The separate token counts of the input (prompt_token_count), the cached content (cached_content_token_count) and the output (candidates_token_count)
    The token count for the thinking process (thoughts_token_count)
    The total number of tokens in both the input and the output (total_token_count)

To understand how big your next conversational turn will be, you need to append it to the history when you call count_tokens.
Python
JavaScript
Go

from google import genai
from google.genai import types

client = genai.Client()

chat = client.chats.create(
    model="gemini-3-flash-preview",
    history=[
        types.Content(
            role="user", parts=[types.Part(text="Hi my name is Bob")]
        ),
        types.Content(role="model", parts=[types.Part(text="Hi Bob!")]),
    ],
)

print(
    client.models.count_tokens(
        model="gemini-3-flash-preview", contents=chat.get_history()
    )
)

response = chat.send_message(
    message="In one sentence, explain how a computer works to a young child."
)
print(response.usage_metadata)

extra = types.UserContent(
    parts=[
        types.Part(
            text="What is the meaning of life?",
        )
    ]
)
history = [*chat.get_history(), extra]
print(client.models.count_tokens(model="gemini-3-flash-preview", contents=history))

Count multimodal tokens

All input to the Gemini API is tokenized, including text, image files, and other non-text modalities. Note the following high-level key points about tokenization of multimodal input during processing by the Gemini API:

    Image inputs with both dimensions <=384 pixels are counted as 258 tokens. Images larger in one or both dimensions are cropped and scaled as needed into tiles of 768x768 pixels, each counted as 258 tokens.

    Video and audio files are converted to tokens at the following fixed rates: video at 263 tokens per second and audio at 32 tokens per second.

Media resolutions

Gemini 3 Pro and 3 Flash Preview models introduces granular control over multimodal vision processing with the media_resolution parameter. The media_resolution parameter determines the maximum number of tokens allocated per input image or video frame. Higher resolutions improve the model's ability to read fine text or identify small details, but increase token usage and latency.

For more details about the parameter and how it can impact token calculations, see the media resolution guide.
Image files

If you call count_tokens with a text-and-image input, it returns the combined token count of the text and the image in the input only (total_tokens). You can make this call before calling generate_content to check the size of your requests. You can also optionally call count_tokens on the text and the file separately.

Another option is calling generate_content and then using the usage_metadata attribute on the response object to get the following:

    The separate token counts of the input (prompt_token_count), the cached content (cached_content_token_count) and the output (candidates_token_count)
    The token count for the thinking process (thoughts_token_count)
    The total number of tokens in both the input and the output (total_token_count)

Example that uses an uploaded image from the File API:

At least for GEMINI THII is how tokens are counted


## Gemini3-Flash

Input price 	Not available 	$0.25 (text / image / video)
$0.50 (audio)
Output price (including thinking tokens) 	Not available 	$1.50
Context caching price 	Not available 	Same as Standard, Batch pricing not yet implemented
$0.05 (text / image / video)
$0.10 (audio)
$1.00 / 1,000,000 tokens per hour (storage price)
Grounding with Google Search* 	Not available 	1,500 RPD (free), then $14 / 1,000 search queries
Grounding with Google Maps 	Not available 	Not available
Used to improve our products 	Yes 	No