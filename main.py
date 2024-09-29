import cv2
import numpy as np

from picamera2 import Picamera2
from visual_odometry import *

with open("intrinsic.npy", "rb") as f:
    intrinsic = np.load(f)

skip_frames = 2
data_dir = ""
vo = CameraPoses(data_dir, skip_frames, intrinsic)

gt_path = []
estimated_path = []
camera_pose_list = []
start_pose = np.ones((3, 4))
start_translation = np.zeros((3, 1))
start_rotation = np.identity(3)
start_pose = np.concatenate((start_rotation, start_translation), axis = 1)

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main = {"format": "XRGB8888", "size": (640, 480)}))
picam2.start()  

process_frames = False
old_frame = None
new_frame = None
frame_counter = 0

cur_pose = start_pose
  
while True:
    new_frame = picam2.capture_array()
    frame_counter += 1
    start = time.perf_counter()
    
    if process_frames:
        q1, q2 = vo.get_matches(old_frame, new_frame)
        if q1 is not None:
            if len(q1) > 8 and len(q2) > 8:
                transf = vo.get_pose(q1, q2)
                cur_pose = cur_pose @ transf
        
        hom_array = np.array([[0, 0, 0, 1]])
        hom_camera_pose = np.concatenate((cur_pose, hom_array), axis=0)
        camera_pose_list.append(hom_camera_pose)
        estimated_path.append((cur_pose[0, 3], cur_pose[2, 3]))
        
        estimated_camera_pose_x, estimated_camera_pose_y = cur_pose[0, 3], cur_pose[2, 3]

    elif process_frames and ret is False:
        break
    
    old_frame = new_frame
    
    process_frames = True
    
    end = time.perf_counter()
    
    total_time = end - start
    fps = 1 / total_time
    
    cv2.putText(new_frame, f"FPS: {int(fps)}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
    cv2.putText(new_frame, str(np.round(cur_pose[0, 3], 1)), (540, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
    cv2.putText(new_frame, str(np.round(cur_pose[1, 3], 1)), (540, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
    cv2.putText(new_frame, str(np.round(cur_pose[2, 3], 1)), (540, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
    
    cv2.imshow("img", new_frame)
    
for cur_pose in camera_pose_list:
    print(cur_pose[0, 3])
    print(cur_pose[1, 3])
    print(cur_pose[2, 3])
    
cv2.destroyAllWindows()