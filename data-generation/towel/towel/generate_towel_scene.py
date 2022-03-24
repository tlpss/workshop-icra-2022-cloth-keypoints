import argparse
import os
import sys

import airo_blender_toolkit as abt
import bpy
import numpy as np
from mathutils import Color

os.environ["INSIDE_OF_THE_INTERNAL_BLENDER_PYTHON_ENVIRONMENT"] = "1"
import blenderproc as bproc


class Towel(abt.KeypointedObject):
    keypoint_ids = {"corner": [0, 1, 2, 3]}

    def __init__(self, length, width):
        self.width = width
        self.length = length

        mesh = self._create_mesh()
        blender_obj = abt.make_object(name="Towel", mesh=mesh)
        super().__init__(blender_obj, Towel.keypoint_ids)

    def _create_mesh(self):
        width, length = float(self.width), float(self.length)

        vertices = [
            np.array([-width / 2, -length / 2, 0.0]),
            np.array([-width / 2, length / 2, 0.0]),
            np.array([width / 2, length / 2, 0.0]),
            np.array([width / 2, -length / 2, 0.0]),
        ]
        edges = [(0, 1), (1, 2), (2, 3), (3, 4)]
        faces = [(0, 1, 2, 3)]

        return vertices, edges, faces


def generate_scene(seed):
    os.environ["BLENDER_PROC_RANDOM_SEED"] = str(seed)
    os.getenv("BLENDER_PROC_RANDOM_SEED")
    bproc.init()

    root_dir = "/home/tlips/Documents/workshop-icra-2022-cloth-keypoints/data-generation"
    haven_folder = os.path.join(root_dir, "utils", "assets", "haven")

    haven_textures_folder = os.path.join(haven_folder, "textures")
    print(haven_folder)

    ground = bproc.object.create_primitive("PLANE")
    ground.blender_obj.name = "Ground"
    ground.set_scale([12] * 3)
    ground.set_rotation_euler([0, 0, np.random.uniform(0.0, 2 * np.pi)])
    ground_texture = abt.random_texture_name(haven_textures_folder)
    print(ground_texture)
    bproc.api.loader.load_haven_mat(haven_textures_folder, [ground_texture])
    ground_material = bpy.data.materials[ground_texture]
    ground.blender_obj.data.materials.append(ground_material)

    # Temporary way to make textures look smaller
    mesh = ground.blender_obj.data
    uv_layer = mesh.uv_layers.active
    for loop in mesh.loops:
        uv_layer.data[loop.index].uv *= 12

    towel_length = np.random.uniform(0.2, 0.5)
    towel_width = np.random.uniform(0.2, towel_length)
    towel = Towel(towel_length, towel_width)
    towel.set_rotation_euler([0, 0, np.random.uniform(0.0, 2 * np.pi)])
    towel.set_location((0, 0, 0.001))
    towel.persist_transformation_into_mesh()

    towel_color = Color()
    towel_hue = np.random.uniform(0, 1.0)
    towel_saturation = np.random.uniform(0.4, 0.8)
    towel_value = np.random.uniform(0.2, 0.8)
    towel_color.hsv = towel_hue, towel_saturation, towel_value
    towel_material = towel.new_material("Towel")
    towel_material.set_principled_shader_value("Base Color", tuple(towel_color) + (1,))

    camera = bpy.context.scene.camera
    camera_location = bproc.sampler.part_sphere(center=[0, 0, 0], radius=1.25, mode="INTERIOR", dist_above_center=0.75)
    camera_rotation = bproc.python.camera.CameraUtility.rotation_from_forward_vec((0, 0, 0) - camera_location)
    camera_pose = bproc.math.build_transformation_mat(camera_location, camera_rotation)
    bproc.camera.add_camera_pose(camera_pose)

    camera.scale = [0.2] * 3  # blender camera object size (no effect on generated images)
    camera.data.lens = 28  # focal distance [mm] - fov approx

    while True:
        x_shift = float(np.random.uniform(-0.3, 0.3))
        y_shift = float(np.random.uniform(-0.3, 0.3))

        towel.set_location((x_shift, y_shift, 0.001))
        towel.persist_transformation_into_mesh()

        break

        if len(towel.keypoints_2D_visible) < 4:
            towel.set_location((0, 0, 0.001))
            towel.persist_transformation_into_mesh()
            print("resample box position")
        else:
            break

    hdri_path = bproc.loader.get_random_world_background_hdr_img_path_from_haven(haven_folder)
    hdri_rotation = np.random.uniform(0, 2 * np.pi)
    abt.load_hdri(hdri_path, hdri_rotation)

    # shader graph

    tree = bpy.data.materials["Towel"].node_tree
    tree.nodes["Principled BSDF"]
    output_node = tree.nodes["Material Output"]

    do_displacement_noise = True

    if do_displacement_noise:
        noise_texture_node = tree.nodes.new("ShaderNodeTexNoise")
        tree.links.new(noise_texture_node.outputs["Color"], output_node.inputs["Displacement"])
        noise_texture_node.inputs["Roughness"].default_value = np.random.uniform(0.5, 1.0)  # 0.5 - 1.0
        noise_texture_node.inputs["Detail"].default_value = 7.0
        noise_texture_node.inputs["Scale"].default_value = np.random.uniform(0.0, 0.3)

    return towel


if __name__ == "__main__":
    seed = 2022
    if "--" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1 :]
        parser = argparse.ArgumentParser()
        parser.add_argument("seed", type=int)
        args = parser.parse_known_args(argv)[0]
        seed = args.seed
    towel = generate_scene(seed)
    # towel.visualize_keypoints()

    print("done")
