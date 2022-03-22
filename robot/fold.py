import numpy as np
import scipy.spatial.transform as transform
from rtde_control import RTDEControlInterface as RTDEControl


def transformation_matrix_from_position_and_vecs(pos, x, y, z):
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, 0] = x
    transformation_matrix[:3, 1] = y
    transformation_matrix[:3, 2] = z
    transformation_matrix[:3, 3] = pos
    return transformation_matrix


class TowelFold:
    def __init__(self, kp1, kp2, kp3, kp4) -> None:
        # cloth is assumed to be below robot.
        # kp1 is most to the left of the "upper" two
        # others are clockwise

        self.cloth_position = (kp1 + kp2 + kp3 + kp4) / 4
        self.x = ((kp2 - kp1) + (kp3 - kp4)) / 2  # average the two vectors to cope w/ slight non-rectangular cloth
        self.len = np.linalg.norm(self.x)
        self.x /= self.len
        self.z = np.array([0, 0, 1])

        self.y = np.cross(self.z, self.x)
        self.robot_to_cloth_base_transform = transformation_matrix_from_position_and_vecs(
            self.cloth_position, self.x, self.y, self.z
        )

    def fold_pose_in_cloth_frame(self, t):
        assert t <= 1 and t >= 0
        angle = np.pi - t * np.pi
        position = np.array([self.len / 2.4 * np.cos(angle), 0, self.len / 2.5 * np.sin(angle)])
        position[2] += 0.085 / 2 * np.sin(np.pi / 4) - 0.01
        # offset for open gripper
        orientation_angle = -3 * np.pi / 4 - t * np.pi / 4
        x = np.array([np.cos(orientation_angle), 0, np.sin(orientation_angle)])
        x /= np.linalg.norm(x)
        y = np.array([0, 1, 0])

        z = np.cross(x, y)
        return transformation_matrix_from_position_and_vecs(position, x, y, z)

    def pregrasp_pose_in_cloth_frame(self, alpha):
        grasp_pose = self.fold_pose_in_cloth_frame(0)
        pregrasp_pose = grasp_pose
        pregrasp_pose[0, 3] = pregrasp_pose[0, 3] - alpha

        return pregrasp_pose

    @staticmethod
    def pose_to_rotvec_waypoint(pose):
        position = pose[:3, 3]
        rpy = transform.Rotation.from_matrix(pose[:3, :3]).as_rotvec()
        return np.concatenate((position, rpy))


if __name__ == "__main__":

    """test setup for fold sequence."""
    cloth = TowelFold(
        np.array([-0.1, -0.2, 0]), np.array([0.3, -0.2, 0]), np.array([0.3, -0.3, 0]), np.array([-0.1, -0.3, 0])
    )
    print(cloth.pregrasp_pose_in_cloth_frame(0.0))
    print()
    print(cloth.robot_to_cloth_base_transform)
    pregrasp_in_towel = cloth.pregrasp_pose_in_cloth_frame(0.05)
    pregrasp = cloth.robot_to_cloth_base_transform @ pregrasp_in_towel
    print(pregrasp)
    print(cloth.pose_to_rotvec_waypoint(pregrasp))
    pregrasp_rpy = cloth.pose_to_rotvec_waypoint(pregrasp)

    from robotiq2f_tcp import Robotiq2F85TCP

    vel = 0.3
    acc = 0.2
    blend = 0.01
    wps = []
    num_waypoints = 50
    for t in range(0, num_waypoints):
        i = t / num_waypoints
        wp = cloth.pose_to_rotvec_waypoint(cloth.robot_to_cloth_base_transform @ cloth.fold_pose_in_cloth_frame(i))
        wps.append(wp.tolist() + [vel, acc, blend])

    print("external control")
    rtde_c = RTDEControl("10.42.0.162")

    gripper = Robotiq2F85TCP("10.42.0.162")
    gripper.activate_gripper()
    gripper.move_to_position(20, 250, 10)

    rtde_c.moveL(pregrasp_rpy, vel, acc)
    check = input("continue fold? Press Enter")
    rtde_c.moveL(
        cloth.pose_to_rotvec_waypoint(cloth.robot_to_cloth_base_transform @ cloth.fold_pose_in_cloth_frame(0)),
        vel,
        acc,
    )
    gripper.move_to_position(230, 255, 10)
    rtde_c.moveL(wps)

    gripper.move_to_position(20, 255, 10)
    gripper.close()
