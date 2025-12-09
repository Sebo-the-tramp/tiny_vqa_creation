import json
from collections import Counter
from pathlib import Path


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    data_path = base_dir / "json" / "all_objects_data.json"

    with data_path.open() as f:
        objects = json.load(f)

    min_entry = None  # (value, object_key, material_name)
    max_entry = None
    material_counter = Counter()

    for obj_key, payload in objects.items():
        material = payload.get("material") or {}
        youngs = material.get("youngs_modulus_pa") or {}

        min_val = youngs.get("min")
        if min_val is not None:
            if min_entry is None or min_val < min_entry[0]:
                min_entry = (min_val, obj_key, material.get("name"))

        max_val = youngs.get("max")
        if max_val is not None:
            if max_entry is None or max_val > max_entry[0]:
                max_entry = (max_val, obj_key, material.get("name"))

        material_name = material.get("name")
        if material_name:
            material_counter[material_name] += 1

    print("Young's modulus extremes (Pa):")
    if min_entry:
        value, key, material_name = min_entry
        print(f"  Minimum min: {value:.3e} Pa ({key}, material={material_name})")
    else:
        print("  No min values found.")

    if max_entry:
        value, key, material_name = max_entry
        print(f"  Maximum max: {value:.3e} Pa ({key}, material={material_name})")
    else:
        print("  No max values found.")

    print("\nMaterial name distribution (top 10):")
    for material_name, count in material_counter.most_common(10):
        print(f"  {material_name}: {count}")


if __name__ == "__main__":
    main()
