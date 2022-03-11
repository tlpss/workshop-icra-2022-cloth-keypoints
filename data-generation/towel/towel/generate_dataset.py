import argparse
import datetime
import os
import sys

from towel.combine_jsons_to_dataset import combine_jsons_to_dataset
from towel.generate_data_sample import generate_data


def generate_dataset(amount_of_samples, datasets_dir, resolution=256):
    dirname = f"towel n={amount_of_samples} ({datetime.datetime.now()})"
    
    output_dir = os.path.join(datasets_dir, dirname)
    os.makedirs(output_dir)

    for i in range(amount_of_samples):
        generate_data(output_dir, seed=i, resolution=resolution)

    combine_jsons_to_dataset(output_dir)


if __name__ == "__main__":
    if "--" in sys.argv:
        home = os.path.expanduser("~")
        default_datasets_path = os.path.join(home, "datasets")

        argv = sys.argv[sys.argv.index("--") + 1 :]
        parser = argparse.ArgumentParser()
        parser.add_argument("--amount_of_samples", type=int)
        parser.add_argument("--resolution", type=int, default=256)
        parser.add_argument(
            "-d", "--dir", dest="datasets_dir", metavar="DATASETS_DIRECTORY", default=default_datasets_path
        )
        args = parser.parse_known_args(argv)[0]

        generate_dataset(args.amount_of_samples, args.datasets_dir, args.resolution)