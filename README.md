# Sim2Real keypoint detection for Towel folding

This repo contains the code for a submission for the ICRA 2022 workshop on Deformable Object Manipulation.


## Data Generation

To generate synthetic images of unfolded cloths, we make use of [Blender](), version 3.0.
Additionally, we use the excellent [BlenderProc] library and our own [Blender-toolbox]().
### Local Installation
- download blender 3.0
- cd `<path-to-blender>/blender-3.0.1-linux-x64/3.0/python/bin`
- ./python3.9 -m ensurepip
- clone and pip install BlenderProc and the Airo-Blender-Toolkit within the blender python distribution.
- pip install the towel package in this repo in the blender python distribution using `pip install -e ./data-generation/towel`
- run

## Keypoint Detection

## Robot Control
