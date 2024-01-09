# -*- coding: utf-8 -*-
# @Time    : 2023-03-02 11:00
# @Author  : zhangbowen



from utils.hyperparameters import INPUT_SIZE, SEQUENCE_SIZE, BATCH_SIZE, VAL_DATA
import torch
import torchvision
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np




def embedding(data, target):
    '''
    data embedding:
    1. convert data to time-series input format: (num_samples_original, 17, 2) -> (num_samples_time-series, sequence_size, input_size), 
    e.g. (132, 17, 2) -> (33, 1*4, 2*17)    
    
    2. num_samples_original / sequence_size = num_samples_time-series 
    e.g. 132 / 4 = 33
    
    target reduction:
    reduce targets to correspond to time-series input: (num_samples_original,) -> (num_samples_time-series,)
    e.g. (132,) -> (33,)
    '''
    data = data.reshape(-1, INPUT_SIZE) # (132, 17, 2) -> (132, 34)
    data = data.reshape(-1, SEQUENCE_SIZE, data.shape[1]) # (132, 34) -> (33, 4, 34)

    target = target[::SEQUENCE_SIZE] # take out 1 value per SEQUENCE_SIZE (e.g. 4) values
    
    return data, target




########## pre-processing ##########
def pre_processing():
    dataset = pd.read_csv('./data/dataset.csv', index_col=0).to_numpy() # original data

    # data(132, 34), target(132,)
    data = dataset[-133:-1, :-1] # only use 66+66=132 for label balance
    target = dataset[-133:-1, -1]

    # data(132, 17, 2)
    data = data.reshape(-1, 17, 2)
    
    # get center(132, 2)
    left_hip = data[:, 11] # 'x_left_hip' & 'y_left_hip': 11
    right_hip = data[:, 12] # 'x_right_hip' & 'y_right_hip': 12
    center = left_hip * 0.5 + right_hip * 0.5

    # get distance vector(132, 17, 2)
    center = center[:, np.newaxis, :] # (132, 2) -> (132, 1, 2)
    distance = data - center # ndarray automatically broadcase (132, 1, 2) -> (132, 17, 2)

    # normalize(132, 17, 2)
    scale = np.max(abs(distance))
    distance_normalized = distance / scale

    # embedding
    data_embedded, reduced_target = embedding(distance_normalized, target)

    return data_embedded, reduced_target




########## dataset ##########
class DanceDataset():
    def __init__(self, mode):
        self.data, self.target = pre_processing()
        self.mode = mode

    def get_item(self):
        # get data & target (ndarray >> torch float32)
        data = torch.Tensor(self.data) # (33, 4, 34)
        target = torch.Tensor(self.target) # (33)

        # split train & val 
        num_val = int(len(target)*VAL_DATA)
        if self.mode == 'train':
            data = data[:-num_val] # (27, 4, 34)
            target = target[:-num_val] # (27)
        elif self.mode == 'val':
            data = data[-num_val:] # (6, 4, 34)
            target = target[-num_val:] # (6)

        return data, target




########## get_data ##########
def get_data(BATCH_SIZE):
    # train dataset
    data, target = DanceDataset(mode='train').get_item()
    train_dataset = TensorDataset(data, target)

    # val dataset
    data, target = DanceDataset(mode='val').get_item()
    val_dataset = TensorDataset(data, target)

    # train dataloader
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        # num_workers= 2, # accelerate batch->RAM via processor (default:0, recommended: cpu core num)  
    )
    
    # val dataloader
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        # num_workers = 2,
    )

    return train_dataset, val_dataset, train_loader, val_loader



if __name__=='__main__':
    
    # get_data
    train_dataset, val_dataset, train_loader, val_loader = get_data(BATCH_SIZE)
    print('====================')

    # 1 data
    data = train_dataset[0][0]
    target = val_dataset[0][1]
    print('data: {}'.format(data.shape))
    print('target: {}'.format(target))
    print('====================')


    # dataset
    print('train dataset: {}'.format(len(train_dataset)))
    print('val dataset: {}'.format(len(val_dataset)))
    print('====================')

    # dataloader (batch size)
    for check_iteration, (check_datas, check_targets) in enumerate(train_loader):
        print('check batch')
        print('iteration: {}'.format(check_iteration)) 
        print('batch images: {}'.format(check_datas.shape))
        print('batch labels: {}'.format(check_targets.shape))
        print('====================')
        break