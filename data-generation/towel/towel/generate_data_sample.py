import argparse
import datetime
import json
import os
import sys

import bpy
from towel.generate_towel_scene import generate_scene


def generate_data(output_dir, seed, resolution=256):
    print("test")
    towel = generate_scene(seed)
    scene = bpy.context.scene
    print("scene generated")

    image_name = f"{str(seed)}.png"
    image_path_relative = os.path.join("images", image_name)
    image_path = os.path.join(output_dir, image_path_relative)

    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution

    scene.render.filepath = image_path

    data = {
        "image_path": image_path_relative,
    }

    keypoints_2D = towel.json_ready_keypoints(dimension=2, only_visible=True)
    keypoints_2D_visible = towel.json_ready_keypoints(dimension=2, only_visible=False)

    data = data | keypoints_2D | keypoints_2D_visible

    # Saving the data as json
    json_dir = os.path.join(output_dir, "json")
    os.makedirs(json_dir, exist_ok=True)
    json_path = os.path.join(json_dir, f"{str(seed)}.json")
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

    bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    if "--" in sys.argv:

        argv = sys.argv[sys.argv.index("--") + 1 :]
        parser = argparse.ArgumentParser()
        parser.add_argument("seed", type=int)
        parser.add_argument("--resolution", type=int, default=256)
        parser.add_argument("-d", "--dir", dest="datasets_dir", default="./datasets/")  # blender folder/datasets
        args = parser.parse_known_args(argv)[0]

        output_dir = os.path.join(args.datasets_dir, f"towel_{datetime.datetime.now()}")

        generate_data(output_dir, args.seed, args.resolution)
