# import json

# with open("simple_vqa.json") as f:
#     data = json.load(f)

# print(f"Loaded {len(data)} questions.")

# sub_categories = set()
# for categories in data:
#     print(categories)
#     for question in data[categories]:
#         print(question)
#         if "sub_category" in data[categories][question]:
#             sub_categories.add(data[categories][question]["sub_category"])

# print("Sub-categories found:")
# for sub_cat in (sub_categories):
#     print(f"\"{sub_cat}\": \"\",")


import json

with open("./answering_questions/balancing_sub_categories.json") as f:
    data = json.load(f)
print(f"Loaded {len(data)} questions.")

total = 0
for question in data:
    print(data[question])
    total += data[question]

print(f"Total: {total}")
if total != 1.0:
    print("Percentages do not sum to 1.0!")
