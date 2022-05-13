from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms
import wandb  # noqa
from camera_toolkit.zed2i import Zed2i
from keypoint_detection.models.detector import KeypointDetector
from keypoint_detection.utils.heatmap import get_keypoints_from_heatmap
from manual_keypoints import fold_cloth


def crop(img_batch, start_v, height, start_u, width):
    return img_batch[:, :, start_v : start_v + height, start_u : start_u + width]


def uncrop_keypoints(keypoints, start_v, height, start_u, width, img_size):
    # keypoints are now in (U,V) coordinates instead of (V,U) (HxW)
    keypoints[:, 0] = keypoints[:, 0] * width / img_size + start_u
    keypoints[:, 1] = keypoints[:, 1] * height / img_size + start_v
    return keypoints


def get_ordered_keypoints(keypoints):
    """orders the 4 keypoints clockwise starting with the top left keypoint.

    Args:
        keypoints

    Returns: 2D np array with 4 keypoints.

    """
    if isinstance(keypoints, list):
        keypoints = np.array(keypoints)

    dst = np.linalg.norm(keypoints, axis=1)

    kp1 = keypoints[np.argmin(dst)]
    kp3 = keypoints[np.argmax(dst)]
    remaining_keypoints = np.delete(keypoints, [np.argmin(dst), np.argmax(dst)], 0)

    kp2 = remaining_keypoints[np.argmax(remaining_keypoints[:, 0])]

    kp4 = remaining_keypoints[1 - np.argmax(remaining_keypoints[:, 0])]

    return np.array([kp1, kp2, kp3, kp4])


if __name__ == "__main__":
    """
    1. capture image and crop & resize
    2. do inference to get keypoints
    3. transform keypoints to original image frame
    4. execute fold
    """

    crop_u_start = 300
    crop_u_width = 500
    crop_v_start = 250
    crop_v_height = 350
    network_image_size = 256

    # ## Get Model checkpoint from wandb

    # checkpoint_reference = "tlips/icra-2022-workshop/model-2qand21y:v11"
    # # download checkpoint locally (if not already cached)
    # run = wandb.init(project="test-project", entity="tlips")
    # artifact = run.use_artifact(checkpoint_reference, type="model")
    # artifact_dir = artifact.download()
    # load checkpoint
    # model = KeypointDetector.load_from_checkpoint(
    #     Path(artifact_dir) / "model.ckpt", map_location="cpu", backbone_type="Unet"
    # )

    # load local checkpoint
    model = KeypointDetector.load_from_checkpoint(
        Path(__file__).parent / "model.ckpt", map_location="cpu", backbone_type="Unet"
    )
    zed = Zed2i()
    cam_matrix = zed.get_mono_camera_matrix()
    resize_transform = torchvision.transforms.Resize((network_image_size, network_image_size))
    pil_to_torch_transform = torchvision.transforms.ToTensor()

    while True:
        towel_id = 0
        # towel_id = input("Towel ID")
        image_base_name = Path(__file__).parent / "evaluation/" / f"{datetime.now()}"
        # capture image
        img = zed.get_mono_rgb_image()  # CxWxH
        img = zed.image_shape_torch_to_opencv(img)  # WxHxC

        img = img[:, :, [2, 1, 0]]  # BGR to RGB

        # crop and transform
        img = pil_to_torch_transform(img)
        img = torch.unsqueeze(img, 0)
        img = crop(img, crop_v_start, crop_v_height, crop_u_start, crop_u_width)
        img = resize_transform(img)

        # get heatmaps and extract keypoints
        heatmaps = model(img).detach()
        heatmap = heatmaps[:, 0]
        n_keypoints = 4
        keypoints = get_keypoints_from_heatmap(heatmap[0].cpu(), 25, n_keypoints)
        print(keypoints)

        # display keypoints and save w/ and w/o keypoints
        cv_image = torchvision.transforms.ToPILImage()(img[0])
        cv_image = cv2.cvtColor(np.array(cv_image), cv2.COLOR_RGB2BGR)
        cv_image = cv2.resize(cv_image, (crop_u_width, crop_v_height))
        cv2.imwrite(str(image_base_name) + "_initial_state.png", cv_image)

        for keypoint in keypoints:
            # rescale keypoints to match image in original aspect ratio
            rescaled_x = int(keypoint[0] * crop_u_width / 256)
            rescaled_y = int(keypoint[1] * crop_v_height / 256)
            cv2.circle(cv_image, (rescaled_x, rescaled_y), 5, (0, 255, 255), -1)

        cv2.imwrite(str(image_base_name) + "_keypoints.png", cv_image)
        cv2.imshow("keypoints", cv_image)
        key = cv2.waitKey()
        if key == ord("q"):  # do this to abort current fold attempt
            print("aborting")
            continue

        # transform keypoints to original image frame
        if len(keypoints) != 4:
            print("keypoint detection failed, no 4 keypoints found...")
        else:
            sorted_keypoints = get_ordered_keypoints(keypoints)
            print(f"sorted keypoints: {sorted_keypoints}")
            uncropped_keypoints = uncrop_keypoints(sorted_keypoints, 250, 350, 300, 500, 256)
            print(f"{uncropped_keypoints=}")
            fold_cloth(uncropped_keypoints, zed, ask_before_fold=False)

            # get post fold image

            img = zed.get_mono_rgb_image()  # CxWxH
            img = zed.image_shape_torch_to_opencv(img)  # WxHxC
            img = img[:, :, [2, 1, 0]]  # BGR to RGB
            # crop and transform
            img = pil_to_torch_transform(img)
            img = torch.unsqueeze(img, 0)
            img = crop(img, crop_v_start, crop_v_height, crop_u_start, crop_u_width)
            img = resize_transform(img)
            cv_image = torchvision.transforms.ToPILImage()(img[0])
            cv_image = cv2.cvtColor(np.array(cv_image), cv2.COLOR_RGB2BGR)
            cv_image = cv2.resize(cv_image, (crop_u_width, crop_v_height))

            cv2.imwrite(str(image_base_name) + "_post_fold.png", cv_image)
            cv2.imshow("post fold", cv_image)
            key = cv2.waitKey()
            if key == ord("q"):
                break
        cv2.destroyAllWindows()

    cv2.destroyAllWindows()
    zed.close()
