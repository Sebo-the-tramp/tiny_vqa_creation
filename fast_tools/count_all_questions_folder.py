import os
import json

folders = [
    "/data0/sebastian.cavada/compositional-physics/tiny_vqa_deterministic/output/run_07_general",
    "/data0/sebastian.cavada/compositional-physics/tiny_vqa_deterministic/output/run_07_soft",
    "/data0/sebastian.cavada/compositional-physics/tiny_vqa_deterministic/output/run_07_medium",
    "/data0/sebastian.cavada/compositional-physics/tiny_vqa_deterministic/output/run_07_stiff"
]

overall_total = 0

categories_count = {}
sub_categories_count = {}

for folder in folders:
    total_questions = 0
    for file_name in os.listdir(folder):

        # script_final_part = folder.split("/")[-1].replace("run_", "")+".json"
        script_final_part_1K = folder.split("/")[-1].replace("run_", "")+"_1K.json"
        script_final_part_10K = folder.split("/")[-1].replace("run_", "")+"_10K.json"      

        print(script_final_part_1K)

        if file_name.endswith(script_final_part_1K) and "test" in file_name:
            file_path = os.path.join(folder, file_name)
            with open(file_path, 'r') as f:
                data = json.load(f)
                num_questions = len(data)
                total_questions += num_questions
                print(f"File: {file_name}, Questions: {num_questions}")

                # Count categories and subcategories
                for item in data:
                    category = item.get("category")
                    sub_category = item.get("sub_category")
                    if category:
                        categories_count[category] = categories_count.get(category, 0) + 1
                    if sub_category:
                        sub_categories_count[sub_category] = sub_categories_count.get(sub_category, 0) + 1

        if file_name.endswith(script_final_part_10K) and "test" in file_name:
            file_path = os.path.join(folder, file_name)
            with open(file_path, 'r') as f:
                data = json.load(f)
                num_questions = len(data)
                total_questions += num_questions
                print(f"File: {file_name}, Questions: {num_questions}")

                # Count categories and subcategories
                for item in data:
                    category = item.get("category")
                    sub_category = item.get("sub_category")
                    if category:
                        categories_count[category] = categories_count.get(category, 0) + 1
                    if sub_category:
                        sub_categories_count[sub_category] = sub_categories_count.get(sub_category, 0) + 1

    print(f"Total questions in folder {folder}: {total_questions}")

    overall_total += total_questions

print(f"Overall total questions across all folders: {overall_total}")

# Print category and subcategory counts
print("\nCategory counts:")
for category, count in categories_count.items():
    print(f"  {category}: {count}")

total = 1000+1000+1000+10000
# total = 10000
print(f"\nOverall total used for percentage calculations: {total}")

print("\nSubcategory counts:")
for sub_category, count in sub_categories_count.items():
    print(f"  {sub_category}")
    print(f"  {count/total:.2f}")