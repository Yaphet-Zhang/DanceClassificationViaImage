# -*- coding: utf-8 -*-
# @Time    : 2023-03-13 09:20
# @Author  : zhangbowen


import torch
import cv2
import numpy as np
from utils.hyperparameters import INPUT_SIZE_1, COLOR_MEAN, COLOR_STD
from utils.hyperparameters import BATCH_SIZE, NAME_CLASSES, EPOCH, NAME_NET, SEQUENCE_SIZE, INPUT_SIZE
from model.lstm import LSTM
from model.openpose import OpenPoseNet
from dataset import get_data
from utils.decode_pose import decode_pose
from utils.utils import keypoint_18, keypoint_17
import warnings
warnings.filterwarnings('ignore')


########## device ##########
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if str(device) == 'cuda:0':
    print('total gpu(s): {}'.format(torch.cuda.device_count()))
    print('gpu name(s): {}'.format(torch.cuda.get_device_name()))
    print('gpu spec(s): {} GB'.format(torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024))
print('device: {}'.format(device))
print('====================')


########## check camera ##########
# cap = cv2.VideoCapture(r'./custom_no_dance/video/no_dance_zhang1.mp4')
# cap = cv2.VideoCapture(r'./AIST_dance/video/gBR_sBM_c01_d04_mBR0_ch01.mp4')
cap = cv2.VideoCapture(0)
if cap.isOpened():
    print('camera: opened')
    print('camera speed:', cap.get(cv2.CAP_PROP_FPS), 'fps')
    print('frame shape (W, H):', cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
else:
    print('camera: not opened')
print('====================')


################################################################################################
#################### step 1: image -> keypoint coordinate (pose estimation) ####################
################################################################################################


########## network 1 ##########
net_pose = OpenPoseNet().to(device)
print('network pose:', net_pose)
print('====================')

########## load weights 1 ##########
weights_pose_path = './weights/pose_model_scratch.pth'
net_pose_weights = torch.load(weights_pose_path, map_location=device)
# because, pre-trained model (layer names) is different from our model (layer names)
# so, copy pre-trained model (layer names) to our model (layer names: net.state_dict().keys)
keys = list(net_pose_weights.keys())
weights_load = {}
for i in range(len(keys)):
    weights_load[list(net_pose.state_dict().keys())[i]] = net_pose_weights[list(keys)[i]]
state = net_pose.state_dict()
state.update(weights_load)
flag = net_pose.load_state_dict(state)
# print(flag)
print(weights_pose_path + ': loaded!')
print('====================')


########################################################################################################################################
#################### step 2: keypoint coordinate (preprocessing) -> distance vector (time-series model) -> category ####################
########################################################################################################################################


########## network 2 ##########
net_category = LSTM().to(device)
print('network category:', net_category)
print('====================')


########## load weights 2 ##########
weights_category_path = 'weights/' + NAME_NET + str(EPOCH) + '.pth'
flag = net_category.load_state_dict(torch.load(weights_category_path, map_location=device))
# print(flag)
print(weights_category_path + ': loaded!')
print('====================')


###############################
########## real-time ##########
###############################

stack_len = 20 # (x, y, score, joint name), maximum of 1 stack: 10s (10*2=20 frames) when using 2 frames per second
stacked_results = [] 
p_to_pp_threshold = 10 # frame number threshold of p -> pp: at least 5s, that means at least 5*2 = 10 frames
num_frame = 0
frame_interval = 5 
data = []

while(True):
    _, frame = cap.read()
    print('frame num:', num_frame)
    print('====================')

    if num_frame % frame_interval == 0: # only use 1 frame per 0.5s (per 15 frames)


        ################################################################################################
        #################### step 1: image -> keypoint coordinate (pose estimation) ####################
        ################################################################################################

        ########## pre-processing ##########
        oriImg = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # BGR -> RGB
        img = cv2.resize(oriImg, INPUT_SIZE_1, interpolation=cv2.INTER_CUBIC) # resize
        img = img.astype(np.float32) / 255. # normalization
        preprocessed_img = img.copy() # RGB
        for i in range(3): # standardization
            preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - COLOR_MEAN[i]
            preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / COLOR_STD[i]
        img = preprocessed_img.transpose((2, 0, 1)).astype(np.float32) # (H, W, C) -> (C, H, W)
        img = torch.from_numpy(img) # ndarray -> tensor
        x = img.unsqueeze(0).to(device) # (C, H, W) -> (N, C, H, W):(1, 3, 368, 368)


        ########## inference ##########
        with torch.no_grad():
            # forward (output heatmaps & PAFs)
            net_pose.eval()
            predicted_outputs, _ = net_pose(x) # _ is saved_for_loss
            # tensor -> ndarray
            pafs = predicted_outputs[0][0].to('cpu').detach().numpy().transpose(1, 2, 0)
            heatmaps = predicted_outputs[1][0].to('cpu').detach().numpy().transpose(1, 2, 0)
            # resize to input size of openpose
            pafs = cv2.resize(pafs, INPUT_SIZE_1, interpolation=cv2.INTER_CUBIC)
            heatmaps = cv2.resize(heatmaps, INPUT_SIZE_1, interpolation=cv2.INTER_CUBIC)
            # resize to original
            pafs = cv2.resize(pafs, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)
            heatmaps = cv2.resize(heatmaps, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)


        ########## decode heatmap & paf to keypoint ##########
        # joint_list (e.g. 18, 5): (row: number of joints), (column: x, y, score, id, joint type)
        _, result_img, joint_list, _ = decode_pose(oriImg, heatmaps, pafs)
        # original_img, result_img, joint_list, person_to_joint_assoc = decode_pose(oriImg, heatmaps, pafs)


        ########## viz raw pose result ##########
        cv2.imshow('raw pose result', result_img[:, :, ::-1])


        ########## post-processing ##########
        # get all keypoint_18_format results except 'neck'
        result_keypoint_17 = []
        for i in range(len(joint_list)):
            if keypoint_18[joint_list[i][-1]] != 'neck': # delet 'neck'
                # print('x:{}  ||  y:{}  ||  score:{:.2f}  ||  joint:{} ({})'.format(
                #     joint_list[i][0], joint_list[i][1], joint_list[i][2], int(joint_list[i][-1]), keypoint_18[joint_list[i][-1]]
                # ))
                # (x, y, score, joint name)
                result_keypoint_17.append([ joint_list[i][0], joint_list[i][1], joint_list[i][2], keypoint_18[joint_list[i][-1]] ]) 
        result_keypoint_17 = np.array(result_keypoint_17)
        print('raw pose result', result_keypoint_17.shape)
        print('====================')

        # if not joint detected, skip
        if len(result_keypoint_17) == 0:
            continue

        # process repeated data & missing data
        if_can_get_p_and_pp = True
        for key_name in keypoint_17.values(): # loop*17 to check each keypoint
            index = np.where(result_keypoint_17[:, 3]==key_name)[0] # 3 is joint name
            #############################################
            ##### if repeated, select highest score #####
            #############################################
            if len(index) > 1:
                print('repeat:', key_name, index)

                # highest index
                highest_index = index[np.argmax(result_keypoint_17[index][:, 2])] # 2 is score
                # except highest index 
                delet_index = index[highest_index != index]
                # delet except highest index
                result_keypoint_17 = np.delete(result_keypoint_17, delet_index, axis=0) # 0 is row
                # print(result_keypoint_17)
                print('====================')
            #################################
            ##### if missing, add value #####
            #################################
            elif len(index) < 1:
                print('miss:', key_name, index)

                # find missing p & pp
                flag_p_pose = 'none'
                flag_pp_pose = 'none'
                p_pose = None
                pp_pose = None
                for p_to_pp_conunt, result in enumerate(reversed(stacked_results)): # reversed traversal
                    # get missing keypoint's index of each previous result
                    p_or_pp_index = np.where(result[:, 3] == key_name)[0] # 3 is joint name
                    # find p (once p is present, use it and break)
                    if flag_p_pose == 'none' and len(p_or_pp_index) != 0:
                        p_pose = result[p_or_pp_index][0]
                        t_p = p_to_pp_conunt + 1
                        flag_p_pose = 'done'
                    # find pp (once pp is present, use it and break)
                    if flag_pp_pose == 'none' and len(p_or_pp_index) != 0 and p_to_pp_conunt >= p_to_pp_threshold:
                        pp_pose = result[p_or_pp_index][0]
                        t_pp = p_to_pp_conunt + 1
                        flag_pp_pose = 'done'

                # add p & pp
                if p_pose is not None and pp_pose is None: # if get only p
                    c_pose = p_pose # (x, y, score, joint name)     
                    result_keypoint_17 = np.vstack((c_pose, result_keypoint_17))
                    # print(result_keypoint_17)
                elif p_pose is not None and pp_pose is not None: # if get p & pp
                    p = p_pose[0:2].astype(np.float32) # (x, y)
                    pp = pp_pose[0:2].astype(np.float32)
                    # ((t_c-t_pp) / (2*t_c-t_p-t_pp)) * pose_p + ((t_c-t_p) / (2*t_c-t_p-t_pp)) * pose_pp
                    c = ((0-t_pp) / (2*0-t_p-t_pp)) * p + ((0-t_p) / (2*0-t_p-t_pp)) * pp  
                    c_pose = np.hstack((c, p_pose[2:4])) # concat (x, y) & (score, joint name)
                    result_keypoint_17 = np.vstack((c_pose, result_keypoint_17))
                    # print(result_keypoint_17)
                else: # if cannot get p & pp, continue next frame
                    if_can_get_p_and_pp = False
                    # print(result_keypoint_17)
                    break
                print('====================')
        print('processed pose result', result_keypoint_17.shape)


        ########## add result to stacked list after processing repeated data & missing data ##########
        if len(stacked_results) < stack_len:
            stacked_results.append(result_keypoint_17) 
        else:
            stacked_results = [] # initial
            stacked_results.append(result_keypoint_17) 
        print('stack len:', len(stacked_results))
        print('====================')


        ########## viz processed pose result ##########
        for x_, y_, s_, j_  in result_keypoint_17:
            x_ = int(x_.astype(np.float32))
            y_ = int(y_.astype(np.float32))
            cv2.circle(frame, (x_, y_), 3, color=(0, 0, 255), thickness=-1) # (x, y)
            cv2.putText(frame, j_, (x_+5, y_-5), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 0), 2) # joint name


        # check whether keypoint is added to 17, if not (cannot get 'p' or 'p&pp'), continue next frame 
        if if_can_get_p_and_pp:
            cv2.putText(frame, 'VALID', (int(oriImg.shape[1]/2), int(oriImg.shape[0]/6)), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2) # joint name
            cv2.imshow('processed pose result', frame)
        else: 
            cv2.putText(frame, 'INVALID', (int(oriImg.shape[1]/2), int(oriImg.shape[0]/6)), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 2) # joint name
            cv2.imshow('processed pose result', frame)
            print()
            print()
            num_frame += 1
            continue



        ########################################################################################################################################
        #################### step 2: keypoint coordinate (preprocessing) -> distance vector (time-series model) -> category ####################
        ########################################################################################################################################


        ########## pre-processing ##########
        # sort keypoints
        sorted_result_keypoint_17 = []
        for key_name in keypoint_17.values(): # loop*17 to check each keypoint
            index_ = np.where(result_keypoint_17[:, 3]==key_name)[0] # 3 is joint name
            sorted_result_keypoint_17.append(result_keypoint_17[index_][0][0:2].astype(np.float32))
        sorted_result_keypoint_17 = np.array(sorted_result_keypoint_17)

        # stack for time-series
        data.append(sorted_result_keypoint_17)
        if len(data) == SEQUENCE_SIZE:
            # (sequence_size, 17, 2)
            data = np.array(data) 
            print('stacked results:', data.shape)
            print('====================')
            # get center (sequence_size, 2)
            left_hip = data[:, 11] # 'x_left_hip' & 'y_left_hip': 11
            right_hip = data[:, 12] # 'x_right_hip' & 'y_right_hip': 12
            center = left_hip * 0.5 + right_hip * 0.5
            # get distance vector (sequence_size, 17, 2)
            center = center[:, np.newaxis, :] # (sequence_size, 2) -> (sequence_size, 1, 2)
            distance = data - center # ndarray automatically broadcase (sequence_size, 1, 2) -> (sequence_size, 17, 2)
            # normalization (sequence_size, 17, 2)
            scale = np.max(abs(distance))
            distance_normalized = distance / scale
            # embedding (sequence_size, 17, 2) -> (1 batch, sequence_size, input_size)
            data_embedded = distance_normalized.reshape(-1, INPUT_SIZE) # (sequence_size, 17, 2) -> (sequence_size, 34)
            data_embedded = data_embedded.reshape(-1, SEQUENCE_SIZE, data_embedded.shape[1]) # (sequence_size, 34) -> (1 batch, 4, 34)
            x = torch.from_numpy(data_embedded).to(device) # ndarray -> tensor
            print('time-series:', x.shape)
            print('====================')


            ########## inference ##########
            with torch.no_grad():
                # forward (output dance or no-dance)
                net_category.eval()
                pred = net_category(x)
                # max logit >> prediction
                prediction = torch.argmax(pred).item()
                print('prediction: {}:{}'.format(prediction, NAME_CLASSES[prediction]))
                print('====================')
            data = [] # initial

            # viz category result
            if prediction == 0:
                cv2.putText(frame, 'No-Dance', (int(oriImg.shape[1]/2), int(oriImg.shape[0]*0.8)), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2)
            elif prediction == 1:
                cv2.putText(frame, 'Dance', (int(oriImg.shape[1]/2), int(oriImg.shape[0]*0.8)), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imshow('processed pose result', frame)


    print()
    print()
    num_frame += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

