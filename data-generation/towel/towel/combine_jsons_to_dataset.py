import glob
import json
import os


def combine_jsons_to_dataset(output_dir):
    dataset = []
    for file in glob.glob(os.path.join(output_dir, "json/*.json")):
        with open(file) as f:
            dataset.append(json.load(f))

    dataset_json = {"dataset": dataset}

    with open(os.path.join(output_dir, "dataset.json"), "w") as f:
        json.dump(dataset_json, f, indent=2)


if __name__ == "__main__":
    output_dir = os.getcwd()
    combine_jsons_to_dataset(output_dir)
