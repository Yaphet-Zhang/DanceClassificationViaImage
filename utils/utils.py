import numpy as np
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix


########## normalization ##########
def normalization(data):
    scale = np.max(abs(data))
    return data/scale


########## standardization ##########
standard_scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
standardization = standard_scaler.fit_transform


########## visualization ##########
def visual_CM(y_true, y_pred, name_classes, normalized=False):
    '''
    2 or multi classification
    '''
    # confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # normalized
    cm_normalized = confusion_matrix(y_true, y_pred, normalize='true')
    cm_normalized = np.around(cm_normalized, 2)
    plt.figure('confusion matrix')
    plt.xticks(range(len(cm)), name_classes)
    plt.yticks(range(len(cm)), name_classes)
    plt.title('confusion matrix')
    plt.xlabel('predicted')
    plt.ylabel('true')
    if normalized:
        plt.imshow(cm_normalized, cmap=plt.cm.Blues)
        # write number
        for i in range(len(cm_normalized)):
            for j in range(len(cm_normalized[i])):
                plt.text(i-0.3, j, cm_normalized[j][i])
    else:
        plt.imshow(cm, cmap=plt.cm.Blues)
        # write number
        for i in range(len(cm)):
            for j in range(len(cm[i])):
                plt.text(i-0.3, j, cm[j][i])        
    plt.colorbar()
    plt.show()


def visual_CM_2class(y_true, y_pred):
    '''
    only 2-classification
    '''
    tp = np.sum((np.array(y_true)==1) & (np.array(y_pred)==1))
    tn = np.sum((np.array(y_true)==0) & (np.array(y_pred)==0))
    fp = np.sum((np.array(y_true)==0) & (np.array(y_pred)==1))
    fn = np.sum((np.array(y_true)==1) & (np.array(y_pred)==0))
    confusion_matrix = np.array([[tp, fp],
                                [fn, tn]])
    # heatmap
    plt.imshow(confusion_matrix, cmap=plt.cm.Blues)
    indices = range(len(confusion_matrix))
    plt.xticks(indices, ['1:dance', '0:no-dance'])
    plt.yticks(indices, ['1:dance', '0:no-dance'])
    plt.colorbar()
    plt.title('confusion matrix')
    plt.xlabel('True')
    plt.ylabel('Predict')
    # show
    for first_index in range(len(confusion_matrix)): # row
        for second_index in range(len(confusion_matrix[first_index])): # column
            plt.text(first_index, second_index, confusion_matrix[second_index][first_index])
    plt.show()


########## COCO keypoints format ##########
# order of openpose: 18 keypoints (include 'neck')
keypoint_18 = {
    0: 'nose', 1: 'neck', 2: 'right_shoulder', 3: 'right_elbow', 4: 'right_wrist',
    5: 'left_shoulder', 6: 'left_elbow', 7: 'left_wrist', 8: 'right_hip', 9: 'right_knee', 
    10: 'right_ankle', 11: 'left_hip', 12: 'left_knee', 13: 'left_ankle', 14: 'right_eye', 
    15: 'left_eye', 16: 'right_ear', 17: 'left_ear'
}

# order of AIST & custom dataset: 17 keypoints (not include 'neck')
keypoint_17 = {
    0: 'nose', 1: 'left_eye', 2: 'right_eye', 3: 'left_ear', 4: 'right_ear', 
    5: 'left_shoulder', 6: 'right_shoulder', 7: 'left_elbow', 8: 'right_elbow', 9: 'left_wrist', 
    10: 'right_wrist', 11: 'left_hip', 12: 'right_hip', 13: 'left_knee', 14: 'right_knee', 
    15: 'left_ankle', 16: 'right_ankle'
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