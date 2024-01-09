# -*- coding: utf-8 -*-
# @Time    : 2023-03-02 11:00
# @Author  : zhangbowen


import os
import cv2
import numpy as np
import pickle
import pandas as pd


########## hyperparameters ##########
FPS = 60 # video FPS
shortest_time = 8 # shortest video time(s)
freq = 60 # use 1 image per 60 frames
save_img_from_video = False
id_camera_anno_2d = 0 # use number 0 camera annotation which is in front of person. if calibration, please use others (1, 2, ..., 8). total is 9. 





'''
########## COCO keypoints format ##########
# 17 keypoints 
keypoint_17 = {
    0: 'nose', 1: 'left_eye', 2: 'right_eye', 3: 'left_ear', 4: 'right_ear', 
    5: 'left_shoulder', 6: 'right_shoulder', 7: 'left_elbow', 8: 'right_elbow', 9: 'left_wrist', 
    10: 'right_wrist', 11: 'left_hip', 12: 'right_hip', 13: 'left_knee', 14: 'right_knee', 
    15: 'left_ankle', 16: 'right_ankle'
}

# 18 keypoints (include 'neck')
keypoint_18 = {
    0: 'nose', 1: 'neck', 2: 'right_shoulder', 3: 'right_elbow', 4: 'right_wrist',
    5: 'left_shoulder', 6: 'left_elbow', 7: 'left_wrist', 8: 'right_hip', 9: 'right_knee', 
    10: 'right_ankle', 11: 'left_hip', 12: 'left_knee', 13: 'left_ankle', 14: 'right_eye', 
    15: 'left_eye', 16: 'right_ear', 17: 'left_ear'
}
'''




########## training dataset (17 keypoints for LSTM) ##########
keypoint_17_columns = [
    'x_nose', 'y_nose', # features
    'x_left_eye', 'y_left_eye', 
    'x_right_eye', 'y_right_eye', 
    'x_left_ear', 'y_left_ear', 
    'x_right_ear', 'y_right_ear', 
    'x_left_shoulder', 'y_left_shoulder', 
    'x_right_shoulder', 'y_right_shoulder', 
    'x_left_elbow', 'y_left_elbow', 
    'x_right_elbow', 'y_right_elbow', 
    'x_left_wrist', 'y_left_wrist', 
    'x_right_wrist', 'y_right_wrist', 
    'x_left_hip', 'y_left_hip', 
    'x_right_hip', 'y_right_hip', 
    'x_left_knee', 'y_left_knee', 
    'x_right_knee', 'y_right_knee', 
    'x_left_ankle', 'y_left_ankle', 
    'x_right_ankle', 'y_right_ankle',
    'if_dance' # label
]




########## data preparation ##########
# video
video_dir_path = r'./AIST_dance/video'
video_file_path = os.listdir(video_dir_path)
# 2D anno
anno_dir_path_2d = r'./AIST_dance/keypoints2d'
anno_file_path_2d = os.listdir(anno_dir_path_2d)
# # 3D anno
# anno_dir_path_3d = r'./AIST_dance/keypoints3d'
# anno_file_path_3d = os.listdir(anno_dir_path_3d)
# image
img_dir_path = r'./AIST_dance/image/'


##### delet short videos for time series
print('====================')
print('deleting the short video ......')
video_times = []
for video_path in os.listdir(video_dir_path):
    video_path = os.path.join(video_dir_path, video_path)

    cap = cv2.VideoCapture(video_path)
    num_frame = cap.get(7) # total frame number
    video_time = num_frame / FPS # total video time
    cap.release()
    cv2.destroyAllWindows()

    if video_time < shortest_time:
        os.remove(video_path)
    else:
        video_times.append(video_time)

video_times = np.array(video_times)
print('shortest video after deleting: {:.2f}(s)'.format(video_times.min()))
print('longest video after deleting: {:.2f}(s)'.format(video_times.max()))


print('====================')
##### delet anno which not exists in video 
video_file_list = []
for video_file in video_file_path:
    video_file_list.append(video_file[:8] + 'cAll' + video_file[11:-4])
print('video num: ', len(os.listdir(video_dir_path)))
# # 3D 
# for anno_file in anno_file_path_3d:
#     if not anno_file[:-4] in video_file_list:
#         os.remove(os.path.join(anno_dir_path_3d, anno_file))
# print('3D anno num: ', len(os.listdir(anno_dir_path_3d)))
# 2D
for anno_file in anno_file_path_2d:
    if not anno_file[:-4] in video_file_list:
        os.remove(os.path.join(anno_dir_path_2d, anno_file))
print('2D anno num: ', len(os.listdir(anno_dir_path_2d)))


##### video to images
print('====================')
if save_img_from_video:
    print('saving image from video ......')
    if len(os.listdir(img_dir_path)) == 0:
        for video_file in video_file_path:
            video_file_abs = os.path.join(video_dir_path, video_file)
            # ffmpeg_video_read(video_file, 60) ##### video (59.x fps) -> video (60 fps)
            cap = cv2.VideoCapture(video_file_abs)
            id_frame = 0
            while(True):
                ret, frame = cap.read()
                if ret == True:
                    if id_frame % freq == 0:
                        img_path = img_dir_path + video_file[:8] + 'cAll' + video_file[11:-4] + '_' + str(id_frame) + '.jpg'
                        cv2.imwrite(img_path, frame) # (1080, 1920, 3)
                    id_frame += 1
                else:
                    break
            cap.release()
            cv2.destroyAllWindows()




########## read HPE anno ##########
print('====================')
# ##### 2D anno
anno_dir_path_2d = r'./AIST_dance/keypoints2d'
anno_file_path_2d = os.listdir(anno_dir_path_2d)
coor_2d = [] # x, y coord
for anno_file in anno_file_path_2d:
    anno_file = os.path.join(anno_dir_path_2d, anno_file)
    with open(anno_file, 'rb') as f: # read pkl
        # (key of anno_2d dict) 'keypoints2d': general
        anno_2d = pickle.load(f) # e.g. (9, 720, 17, 3): (9 cameras, 720 frames per video, 17 keypoints, x + y + confidence score)
        for id_frame in range(anno_2d['keypoints2d'].shape[1]):
          if id_frame % freq == 0 and id_frame < (FPS*shortest_time): # only use 8 frames (0, 60, ..., 420)
            coor_2d.append(anno_2d['keypoints2d'][id_camera_anno_2d][id_frame][:, :2].flatten())

# ##### 3D anno
# anno_dir_path_3d = r'./AIST_dance/keypoints3d'
# anno_file_path_3d = os.listdir(anno_dir_path_3d)
# coor_3d = []
# for anno_file in anno_file_path_3d:
#     anno_file = os.path.join(anno_dir_path_3d, anno_file)
#     with open(anno_file, 'rb') as f: 
#         # (key of anno_3d dict) 'keypoints3d': general, 'keypoints3d_optim': after smoothing
#         anno_3d = pickle.load(f) # e.g. (720, 17, 3): (720 frames per video, 17 keypoints, x + y + z)
#         for id_frame in range(anno_3d['keypoints3d'].shape[0]):
#           if id_frame % freq ==0:
#             coor_3d.append(anno_3d['keypoints3d'][id_frame][:, :2].flatten())
# coor_3d = np.array(coor_3d)
# print('coor_3d: ', coor_3d.shape)





########## write csv ##########
coor_2d = np.array(coor_2d) # data
dance_label = np.ones(coor_2d.shape[0])[:, np.newaxis] # label:1
dance_dataset = np.hstack([ coor_2d, dance_label]) # dataset

print('coor_2d: ', coor_2d.shape)
print('label: ', dance_label.shape)
print('dataset: ', dance_dataset.shape)

df = pd.DataFrame(dance_dataset, columns=keypoint_17_columns)
df.to_csv('./data/dataset_temp.csv', index=True)





########## just try to visualize a image ##########
##### original img
print('====================')
img = cv2.imread(img_dir_path + 'gBR_sBM_cAll_d04_mBR0_ch03_180.jpg')
h = img.shape[0]
w = img.shape[1]

##### 2D anno
with open(anno_dir_path_2d + '/gBR_sBM_cAll_d04_mBR0_ch03.pkl', 'rb') as f:
    anno_2d_sample = pickle.load(f)
    coor_2d_sample = anno_2d_sample['keypoints2d'][id_camera_anno_2d][180][:, :2]
for i, (x, y) in enumerate(coor_2d_sample):
    x = x
    y = y
    cv2.circle(img, (int(x), int(y)), 3, color=(0, 0, 255), thickness=-1) # keypoint
    cv2.putText(img, str(i), (int(x)+5, int(y)-5), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 0), 2) # keypoint id
img = cv2.resize(img, (int(w/2), int(h/2))) # just for show in display
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ##### 3D anno
# for i in range(17):
#   x = (w/2 + coor_3d[0][2*i]) * 1.0
#   y = (h/2 - coor_3d[0][2*i+1]) * 1.0
#   cv2.circle(img, (int(x), int(y)), 10, color=(0, 0, 255), thickness=-1)


