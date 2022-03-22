from camera_toolkit.zed2i import Zed2i
from camera_toolkit.aruco_pose import get_aruco_marker_poses
import cv2
import pickle 

Zed2i.list_camera_serial_numbers()
zed = Zed2i()
img= zed.get_mono_rgb_image()
img = zed.image_shape_torch_to_opencv(img)
cam_matrix =  zed.get_mono_camera_matrix()
print(img.shape)

img, t, r = get_aruco_marker_poses(img, cam_matrix, 0.106,cv2.aruco.DICT_6X6_250, True)
print(t)
print(r)
cv2.imshow(",",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
with open("marker.pickle","wb" ) as f:
    pickle.dump([t[0],r[0]],f)
zed.close()