from openai import OpenAI
client = OpenAI()

batch_id_to_cancel="batch_699f443536148190996d7ad6c38e70b1"
res = client.batches.cancel(batch_id_to_cancel)
print(res)