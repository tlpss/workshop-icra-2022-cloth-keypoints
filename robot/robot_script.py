import pickle

import numpy as np
from camera_toolkit.reproject_to_z_plane import reproject_to_ground_plane
from camera_toolkit.zed2i import Zed2i
from fold_parameterization import TowelFold
from robotiq2f_tcp import Robotiq2F85TCP
from rtde_control import RTDEControlInterface as RTDEControl

# specify aruco position in UR base frame
aruco_in_robot_x = -0.004
aruco_in_robot_y = -0.23
robot_to_aruco_translation = np.array([aruco_in_robot_x, aruco_in_robot_y, 0.0])

# blend for fold trajectory waypoints
blend = 0.01


def fold_cloth(
    image_coords: np.ndarray,
    zed: Zed2i,
    vel: float = 0.3,
    acc: float = 0.2,
    ur_robot_ip="10.42.0.162",
    ask_before_fold=True,
):
    """script that executes a TowelFold based on the given keypoints
    Args:
        image_coords (np.ndarray): 2D np array of 2D coordinates (U,V) of the towel keypoints, ordered clockwise starting from the topleft.
        zed (Zed2i): camera handle
        vel (float, optional): linear velocity for robot motions. Defaults to 0.3.
        acc (float, optional): linear acceleration for robot motion. Defaults to 0.2.
        ur_robot_ip (string, optional): IP address of the robot controller.

    Raises:
        ValueError: _description_
    """

    # load camera to marker transform
    with open("marker.pickle", "rb") as f:
        aruco_in_camera_position, aruco_in_camera_orientation = pickle.load(f)
    print(f"{aruco_in_camera_position=}")
    print(f"{aruco_in_camera_orientation=}")

    # get camera extrinsics transform
    cam_matrix = zed.get_mono_camera_matrix()
    aruco_in_camera_transform = np.eye(4)
    aruco_in_camera_transform[:3, :3] = aruco_in_camera_orientation
    aruco_in_camera_transform[:3, 3] = aruco_in_camera_position

    # transform markers to aruco frame
    aruco_coords = [
        reproject_to_ground_plane(image_coords[i, :], cam_matrix, aruco_in_camera_transform)[0] for i in range(4)
    ]

    # transform to robot frame
    robot_coords = aruco_coords + robot_to_aruco_translation
    print(f"{aruco_coords=}")
    print(f"{robot_coords=}")

    # create fold motion instance
    kp1, kp2, kp3, kp4 = robot_coords
    cloth = TowelFold(kp1, kp2, kp3, kp4)

    # compute fold trajectory
    wps = []
    num_waypoints = 50
    for t in range(0, num_waypoints):
        i = t / num_waypoints
        wp = cloth.homogeneous_pose_to_position_and_rotvec(
            cloth.robot_to_cloth_base_transform @ cloth.fold_pose_in_cloth_frame(i)
        )
        wps.append(wp.tolist() + [vel, acc, blend])

    # connect to robot and gripper
    rtde_c = RTDEControl(ur_robot_ip)
    gripper = Robotiq2F85TCP(ur_robot_ip)
    gripper.activate_gripper()

    ### EXECUTE ROBOT MOTIONS ###

    # move to a safe pose before moving to pregrasp to avoid collisions
    pre_fold_waypoint = [0.13, -0.25, 0.15, 0, 3.14, 0]
    rtde_c.moveL(pre_fold_waypoint, vel, acc)

    # open gripper
    gripper.move_to_position(5, 255, 10)

    # move to pregrasp pose
    pregrasp_in_towel = cloth.pregrasp_pose_in_cloth_frame(0.08)
    pregrasp = cloth.robot_to_cloth_base_transform @ pregrasp_in_towel
    print(pregrasp)
    print(cloth.homogeneous_pose_to_position_and_rotvec(pregrasp))
    pregrasp_pose_rotvec = cloth.homogeneous_pose_to_position_and_rotvec(pregrasp)

    rtde_c.moveL(pregrasp_pose_rotvec, vel, acc)

    if ask_before_fold:
        # check to abort if pregrasp pose is not OK.
        input("continue fold? Press Enter")

    # move to grasp pose
    rtde_c.moveL(
        cloth.homogeneous_pose_to_position_and_rotvec(
            cloth.robot_to_cloth_base_transform @ cloth.fold_pose_in_cloth_frame(0)
        ),
        vel,
        acc,
    )
    # close gripper to grasp cloth
    gripper.move_to_position(230, 255, 255)

    # execute fold trajectory
    rtde_c.moveL(wps)

    # open gripper to release cloth
    gripper.move_to_position(5, 255, 10)

    post_fold_waypoint = cloth.homogeneous_pose_to_position_and_rotvec(
        cloth.robot_to_cloth_base_transform @ cloth.fold_retreat_pose_in_cloth_frame()
    )

    rtde_c.moveL(post_fold_waypoint, vel, acc)
    # move to final pose to get gripper out of the way for the camera
    post_fold_waypoint = [0.13, -0.10, 0.22, 0, 3.14, 0]
    rtde_c.moveL(post_fold_waypoint, vel, acc)
