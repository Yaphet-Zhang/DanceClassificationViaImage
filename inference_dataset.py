# -*- coding: utf-8 -*-
# @Time    : 2023-03-02 11:00
# @Author  : zhangbowen


import torch
from model.lstm import LSTM
from dataset import get_data
from utils.hyperparameters import BATCH_SIZE, NAME_CLASSES, EPOCH, NAME_NET
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


########## network ##########
net = LSTM().to(device)
print(net)
print('====================')


########## load weights ##########
weights_path = 'weights/' + NAME_NET + str(EPOCH) + '.pth'
flag = net.load_state_dict(torch.load(weights_path, map_location=device))
print(flag)
print(weights_path + ': loaded!')
print('====================')



########## get data ##########
train_dataset, val_dataset, train_loader, val_loader = get_data(BATCH_SIZE)
# .__getitem__ can be omitted: val_dataset[0][0]
sample1_data = val_dataset.__getitem__(5)[0]
sample1_target = val_dataset.__getitem__(5)[1]
print('sample 1 data: {}'.format(sample1_data.shape))
print('sample 1 label: {}:{}'.format(int(sample1_target), NAME_CLASSES[int(sample1_target)]))
print('====================')



########## inference ##########
net.eval()
with torch.no_grad():
    sample1_data = sample1_data.unsqueeze(0).to(device) # [sequence_size, input_size] >> [batch_size, sequence_size, input_size]
    sample1_target = sample1_target.to(device)
    # sample1_target = torch.tensor(sample1_target, dtype=torch.int64).to(device)

    # forward
    sample1_output = net(sample1_data)

    # max logit >> prediction
    prediction = torch.argmax(sample1_output).item()

    print('sample 1 prediction: {}:{}'.format(prediction, NAME_CLASSES[prediction]))
    print('====================')