# -*- coding: utf-8 -*-
# @Time    : 2023-03-02 11:00
# @Author  : zhangbowen


import json
import numpy as np
import cv2
import os
import pandas as pd




########## hyperparameters ##########
save_img_from_video = False




keypoint_17 = {
    'nose': 0, 'left_eye': 1, 'right_eye': 2, 'left_ear': 3, 'right_ear': 4, 
    'left_shoulder': 5, 'right_shoulder': 6, 'left_elbow': 7, 'right_elbow': 8, 'left_wrist': 9, 
    'right_wrist': 10, 'left_hip': 11, 'right_hip': 12, 'left_knee': 13, 'right_knee': 14, 
    'left_ankle': 15, 'right_ankle': 16
}



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




########## video 2 image ##########
# img dir
img_dir = r'./custom_no_dance/image/'
# video
video_file_path = r'./custom_no_dance/video/no_dance_zhang1.mp4'
cap = cv2.VideoCapture(video_file_path)
if save_img_from_video:
    print('saving image from video ......')
    id_frame = 0
    while(True):
        ret, frame = cap.read()
        if ret == True:
            if id_frame % 30 == 0:
                resized = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_NEAREST) # (1080, 1920, 3)
                img_path =  img_dir + str(id_frame) + '.jpg'
                cv2.imwrite(img_path, resized)
            id_frame += 1
            # cv2.waitKey(10)
            # if cv2.waitKey(1) == ord('q'):
            #     break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()




########## read zhang-created anno ##########
print('====================')
##### 2D anno
anno_dir_path_2d = r'./custom_no_dance/keypoints2d'
anno_file_path_2d = os.listdir(anno_dir_path_2d)
coor_2ds = [] # x, y coords

for anno_file in anno_file_path_2d:
    anno_file = os.path.join(anno_dir_path_2d, anno_file)
    with open(anno_file) as f:
        anno_2d = json.load(f) # read json
        coor_2d = [] # x, y coord
        for i, joint_name in enumerate(keypoint_17):
            if anno_2d['shapes'][i]['label'] == joint_name:
                coor = anno_2d['shapes'][i]['points'][0]
                coor_2d.append(coor)
        coor_2d = np.array(coor_2d).astype(int).flatten()
    coor_2ds.append(coor_2d)




########## write csv ##########
coor_2ds = np.array(coor_2ds) # data
no_dance_label = np.zeros(coor_2ds.shape[0])[:, np.newaxis] # label:0
no_dance_dataset = np.hstack([ coor_2ds, no_dance_label]) # dataset
df_no_dance = pd.DataFrame(no_dance_dataset, columns=keypoint_17_columns)
print('coor_2d: ', coor_2ds.shape)
print('label: ', no_dance_label.shape)
print('dataset: ', no_dance_dataset.shape)

##### read dance csv (label:1)
df_dance = pd.read_csv(r'./data/dataset_temp.csv', index_col=0)
##### add no-dance data to csv (label:0)
df = pd.concat([df_dance, df_no_dance], ignore_index=True)
df.to_csv('./data/dataset.csv', index=True)




########## just try to visualize a image ##########
img_path = r'./custom_no_dance/image/1530.jpg'
anno_path = r'./custom_no_dance/keypoints2d/1530.json'

##### original img
img = cv2.imread(img_path)
h = img.shape[0]
w = img.shape[1]

##### 2D anno
with open(anno_path) as f:
    anno_2d_sample = json.load(f)
    
    coor_2d_sample = []
    for i, joint_name in enumerate(keypoint_17):
        if anno_2d_sample['shapes'][i]['label'] == joint_name:
            coor = anno_2d_sample['shapes'][i]['points'][0]
            coor_2d_sample.append(coor)
coor_2d_sample = np.array(coor_2d_sample)

for i, (x, y) in enumerate(coor_2d_sample):
    x = x
    y = y
    cv2.circle(img, (int(x), int(y)), 3, color=(0, 0, 255), thickness=-1) # keypoint
    cv2.putText(img, str(i), (int(x)+5, int(y)-5), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 0), 2) # keypoint id
img = cv2.resize(img, (int(w/2), int(h/2))) # just for show in display
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


