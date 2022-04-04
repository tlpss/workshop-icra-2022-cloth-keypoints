import numpy as np
import scipy.spatial.transform as transform


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
        """Parameterization of the fold trajectory
        t = 0 is the grasp pose, t = 1 is the final (release) pose
        """
        assert t <= 1 and t >= 0
        position_angle = np.pi - t * np.pi
        # the radius was manually tuned on a cloth to find a balance between grasp width along the cloth and grasp robustness given the gripper fingers.
        position = np.array(
            [(self.len / 2.2 - 0.02) * np.cos(position_angle), 0, (self.len / 2.2 - 0.02) * np.sin(position_angle)]
        )

        position[2] -= 0.03  # gripper is opened here so point of fingers is now this amount above closed-TCP
        position[2] -= 0.01  # offset of the mounting plate
        position[2] += 0.085 / 2 * np.sin(np.pi / 4)  # want the low finger to touch the table so offset from TCP

        orientation_angle = -3 * np.pi / 4 - t * np.pi / 4 * 1.2
        x = np.array([np.cos(orientation_angle), 0, np.sin(orientation_angle)])
        x /= np.linalg.norm(x)
        y = np.array([0, 1, 0])

        z = np.cross(x, y)
        return transformation_matrix_from_position_and_vecs(position, x, y, z)

    def grasp_pose_in_cloth_frame(self):
        return self.fold_pose_in_cloth_frame(0)

    def pregrasp_pose_in_cloth_frame(self, alpha=0.10):
        grasp_pose = self.fold_pose_in_cloth_frame(0)
        pregrasp_pose = grasp_pose
        # create offset in x-axis for grasp approach (linear motion along +x)
        pregrasp_pose[0, 3] = pregrasp_pose[0, 3] - alpha

        return pregrasp_pose

    def fold_retreat_pose_in_cloth_frame(self):
        pose = self.fold_pose_in_cloth_frame(49 / 50)
        pose[2, 3] += 0.05  # move up
        pose[0, 3] += 0.01
        return pose

    @staticmethod
    def homogeneous_pose_to_position_and_rotvec(pose):
        position = pose[:3, 3]
        rpy = transform.Rotation.from_matrix(pose[:3, :3]).as_rotvec()
        return np.concatenate((position, rpy))
