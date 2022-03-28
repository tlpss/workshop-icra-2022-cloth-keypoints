import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torchvision.transforms
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
import torch
import numpy as np

from keypoint_detection.utils.heatmap import gaussian_heatmap, overlay_image_with_heatmap, generate_keypoints_heatmap, overlay_image_with_heatmap, get_keypoints_from_heatmap
from keypoint_detection.models.detector import KeypointDetector
import wandb
from pathlib import Path
from skimage import io
import torchvision

import pickle

import cv2
import numpy as np
from camera_toolkit.reproject_to_z_plane import reproject_to_ground_plane
from camera_toolkit.zed2i import Zed2i

def crop(img_batch,start_v,height, start_u, width):
    return img_batch[:,:,start_v: start_v +height, start_u: start_u + width]
def imshow(img):
    """
    plot Tensor as image
    images are kept in the [0,1] range, although in theory [-1,1] should be used to whiten..
    """
    np_img = img.numpy()
    # bring (C,W,H) to (W,H,C) dims
    img = np.transpose(np_img, (1,2,0))
    plt.imshow(img)
    plt.show()

## Get Model checkpoint from wandb

checkpoint_reference = 'tlips/icra-2022-workshop/model-2tp1tfxe:v8'

# download checkpoint locally (if not already cached)
run = wandb.init(project="test-project", entity="tlips")
artifact = run.use_artifact(checkpoint_reference, type="model")
artifact_dir = artifact.download()

# 
#checkpoint = torch.load(Path(artifact_dir) / "model.ckpt")
#print(checkpoint["hyper_parameters"])
# load checkpoint
model = KeypointDetector.load_from_checkpoint(Path(artifact_dir) / "model.ckpt",map_location="cpu",backbone_type='Unet')

zed = Zed2i()
cam_matrix = zed.get_mono_camera_matrix()


while (True):
    # capture image and get camera to aruco pose
    img = zed.get_mono_rgb_image()
    img = zed.image_shape_torch_to_opencv(img)

    transform  = torchvision.transforms.Resize((256,256))
    pil_to_torch = torchvision.transforms.ToTensor()

    # crop and transform 
    img = pil_to_torch(img)
    img = torch.unsqueeze(img, 0)
    img = crop(img,250,350,300,500)
    img = transform(img)
    heatmaps = model(img).detach()
    heatmap = heatmaps[:,0]
    n_keypoints = 4
    overlayed_heatmap = pil_to_torch(overlay_image_with_heatmap(img[0], torch.unsqueeze(
                                    generate_keypoints_heatmap(
                                        img.shape[-2:],

                                        get_keypoints_from_heatmap(heatmap[0].cpu(), 1,n_keypoints),
                                        sigma=4,
                                        device = 'cpu'
                                    ),
                                    0,
                                ),0.6))

    imshow(overlayed_heatmap)