import argparse
import os
import random
import sys

import airo_blender_toolkit as abt
import bpy
import numpy as np
from mathutils import Color, Vector

os.environ["INSIDE_OF_THE_INTERNAL_BLENDER_PYTHON_ENVIRONMENT"] = "1"
import blenderproc as bproc


def get_random_filename(filedir):
    onlyfiles = [f for f in os.listdir(filedir) if os.path.isfile(os.path.join(filedir, f))]
    onlyfiles.sort()
    return random.choice(onlyfiles)


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
    print(seed)
    os.environ["BLENDER_PROC_RANDOM_SEED"] = str(seed)
    os.getenv("BLENDER_PROC_RANDOM_SEED")
    bproc.init()

    # renderer settings
    bpy.context.scene.cycles.adaptive_threshold = 0.2
    bpy.context.scene.cycles.use_denoising = False

    root_dir = "/home/tlips/Documents/workshop-icra-2022-cloth-keypoints/data-generation"
    haven_folder = os.path.join(root_dir, "utils", "assets", "haven")
    thingi_folder = os.path.join(root_dir, "utils", "assets", "thingi10")

    haven_textures_folder = os.path.join(haven_folder, "textures")

    # create ground texture plane
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

    # create towel
    towel_length = np.random.uniform(0.4, 0.7)
    towel_width = np.random.uniform(0.2, towel_length)
    towel = Towel(towel_length, towel_width)
    towel.set_rotation_euler([0, 0, np.random.uniform(0.0, 2 * np.pi)])
    # shift towel in view
    x_shift = float(np.random.uniform(-0.1, 0.1))
    y_shift = float(np.random.uniform(-0.1, 0.1))

    towel.set_location((x_shift, y_shift, 0.001))
    towel.persist_transformation_into_mesh()

    color = Color()
    towel_hue = np.random.uniform(0, 1.0)
    towel_saturation = np.random.uniform(0.0, 1.0)
    towel_value = np.random.uniform(0.0, 1.0)
    color.hsv = towel_hue, towel_saturation, towel_value
    towel_material = towel.new_material("Towel")
    towel_material.set_principled_shader_value("Base Color", tuple(color) + (1,))

    # add camera
    camera = bpy.context.scene.camera
    camera_location = bproc.sampler.part_sphere(center=[0, 0, 0], radius=1.0, mode="INTERIOR", dist_above_center=0.85)
    camera_rotation = bproc.python.camera.CameraUtility.rotation_from_forward_vec((0, 0, 0) - camera_location)
    camera_pose = bproc.math.build_transformation_mat(camera_location, camera_rotation)
    bproc.camera.add_camera_pose(camera_pose)

    camera.scale = [0.2] * 3  # blender camera object size (no effect on generated images)
    camera.data.lens = 28  # focal distance [mm] - fov approx

    # add HDRI
    hdri_path = bproc.loader.get_random_world_background_hdr_img_path_from_haven(haven_folder)
    hdri_rotation = np.random.uniform(0, 2 * np.pi)
    abt.load_hdri(hdri_path, hdri_rotation)


    # shader graph
    tree = bpy.data.materials["Towel"].node_tree
    towel_renderer = tree.nodes["Principled BSDF"]
    towel_renderer.inputs["Roughness"].default_value = 1.0

    do_displacement_noise = True
    randomize_ground_base_color = True
    add_random_objects = True

    if do_displacement_noise:
        output_node = tree.nodes["Material Output"]
        noise_texture_node = tree.nodes.new("ShaderNodeTexNoise")
        tree.links.new(noise_texture_node.outputs["Color"], output_node.inputs["Displacement"])
        noise_texture_node.inputs["Roughness"].default_value = np.random.uniform(0.5, 1.0)  # 0.5 - 1.0
        noise_texture_node.inputs["Detail"].default_value = 7.0
        noise_texture_node.inputs["Scale"].default_value = np.random.uniform(0.0, 0.3)

    if randomize_ground_base_color:
        tree = ground_material.node_tree
        render_node = tree.nodes["Principled BSDF"]
        try:
            existing_mix_node = tree.nodes["Mix"]
            additional_mix_node = tree.nodes.new("ShaderNodeMixRGB")
            base_color_node = tree.nodes.new("ShaderNodeRGB")
            tree.links.new(base_color_node.outputs["Color"], additional_mix_node.inputs["Color2"])
            tree.links.new(existing_mix_node.outputs["Color"], additional_mix_node.inputs["Color1"])
            tree.links.new(additional_mix_node.outputs["Color"], render_node.inputs["Base Color"])
            # additional_mix_node.inputs["Fac"] = 0.3
            color = Color()
            hue = np.random.uniform(0, 1.0)
            saturation = np.random.uniform(0.0, 1.0)
            value = np.random.uniform(0.0, 1.0)
            color.hsv = hue, saturation, value
            base_color_node.outputs["Color"].default_value = tuple(color) + (1,)
        except Exception as e:
            # some textures do not have the Mix node as they have no 2 jpg's that are mixed. Leave those as is..
            # example: cobblestones_floor_001
            print(e)

    if add_random_objects:
        n_random_objects = int(np.random.uniform(0, 5.0))
        print(f" {n_random_objects} random objects")
        for i in range(n_random_objects):
            random_object_name = get_random_filename(thingi_folder)
            print(random_object_name)
            bpy.ops.import_mesh.stl(filepath=os.path.join(thingi_folder, random_object_name))
            obj = bpy.context.selected_objects[0]
            bb_vertex = obj.matrix_world @ (Vector(obj.bound_box[6]) - Vector(obj.bound_box[0]))
            bb_vertex = [abs(x) for x in bb_vertex]
            obj.scale = np.array(
                [0.1 / (bb_vertex[0]), 0.1 / (bb_vertex[1]), 0.1 / (bb_vertex[2])]
            ) * np.random.uniform(0.5, 3.0)
            obj.location = (np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5), 0)
            material = bpy.data.materials.new(name=f"object{i}")
            color = Color()
            hue = np.random.uniform(0, 1.0)
            saturation = np.random.uniform(0.0, 1.0)
            value = np.random.uniform(0.0, 1.0)
            color.hsv = hue, saturation, value
            material.diffuse_color = tuple(color) + (1,)
            obj.data.materials.append(material)

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
