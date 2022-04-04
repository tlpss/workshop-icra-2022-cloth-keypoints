import cv2
import numpy as np
from camera_toolkit.zed2i import Zed2i
from robot_script import fold_cloth

if __name__ == "__main__":
    """Script to capture image and select the keypoints manually, to test the control."""
    # opencv mouseclick registration
    clicked_coords = []

    def clicked_callback_cv(event, x, y, flags, param):
        global u_clicked, v_clicked
        if event == cv2.EVENT_LBUTTONDBLCLK:
            print(f"clicked on {x}, {y}")
            clicked_coords.append(np.array([x, y]))

    # open ZED (and assert it is available)
    Zed2i.list_camera_serial_numbers()
    zed = Zed2i()
    cam_matrix = zed.get_mono_camera_matrix()

    # capture image
    img = zed.get_mono_rgb_image()
    img = zed.image_shape_torch_to_opencv(img)

    # mark the keypoints in image plane by clicking
    cv2.imshow("img", img)
    cv2.setMouseCallback("img", clicked_callback_cv)

    while True:
        print("click the 4 keypointsbefore closing, mark them clockwise starting with the end of the fold line.")

        cv2.waitKey(0)
        if len(clicked_coords) > 4:
            raise IndexError("too many keypoint clicked, aborting.")
        elif len(clicked_coords) == 4:
            break

    cv2.destroyAllWindows()

    # once 4 are clicked, call fold function
    fold_cloth(np.array(clicked_coords), zed)

    zed.close()
