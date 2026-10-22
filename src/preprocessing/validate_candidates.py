import json
from pathlib import Path

sample = next(Path("data/raw/spacenet7/sample").iterdir())
label_dir = sample / "labels_match_pix"

def load_labels(month):
    path = next(label_dir.glob(f"*{month}*.geojson"))
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        feature["properties"]["Id"]: feature
        for feature in data["features"]
    }

labels_2018 = load_labels("2018_01")
labels_2019 = load_labels("2019_01")

new_ids = sorted(set(labels_2019) - set(labels_2018))

print("2018 annotations:", len(labels_2018))
print("2019 annotations:", len(labels_2019))
print("2019-only candidate IDs:", len(new_ids))
print("\nFirst 10 candidate IDs:", new_ids[:10])

for building_id in new_ids[:5]:
    feature = labels_2019[building_id]
    geometry = feature["geometry"]
    print(
        f"ID={building_id}, "
        f"geometry={geometry['type']}, "
        f"properties={feature['properties']}"
    )