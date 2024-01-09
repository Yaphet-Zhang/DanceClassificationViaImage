import torch
from dataset import get_data
from utils.hyperparameters import BATCH_SIZE, EPOCH, NAME_NET, NAME_CLASSES
from model.lstm import LSTM
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from utils.utils import visual_CM, visual_CM_2class
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


########## get data ##########
train_dataset, val_dataset, train_loader, val_loader = get_data(BATCH_SIZE)

# dataset
print('train dataset: {}'.format(len(train_dataset)))
print('val dataset: {}'.format(len(val_dataset)))

# dataloader (batch size)
for check_iteration, (check_data, check_label) in enumerate(train_loader):
    print('check batch')
    print('iteration: {}'.format(check_iteration)) 
    print('batch data: {}'.format(check_data.shape))
    print('batch label: {}'.format(check_label.shape))
    print('====================')
    break


########## network ##########
net = LSTM().to(device)
print(net)
print('====================')


########## loss function ##########
criterion = nn.CrossEntropyLoss() 


########## optimization ##########
optimizer = optim.Adam(params=net.parameters())


########## training & validation ##########
## initialize time (epochs)
train_time = 0
val_time = 0

## initialize loss, acc, precision, recall, f1-score, AUC list (epochs)
# loss
train_loss_list = []
val_loss_list = []
# acc
train_acc_list = []
val_acc_list = []
# precision
train_precision_list = []
val_precision_list = []
# recall
train_recall_list = []
val_recall_list = []
# f1-score
train_f1_score_list = []
val_f1_score_list = []


########## epoch ##########
for i in range(EPOCH):
    print('--------------------')
    print('Epoch: {}/{}'.format(i+1, EPOCH))
    
    # initialize loss (1 epoch)
    train_loss = 0
    val_loss = 0

    # # initialize acc (1 epoch)
    # train_acc = 0
    # val_acc = 0

    # initialize labels (1 epoch) 
    train_labels = []
    val_labels = []

    # initialize predictions (1 epoch)
    train_preds = []   
    val_preds = []

    ########## training ##########
    ## train start time (1 epoch)
    train_start_time = time.time()
    print('-----Train mode-----')
    net.train()
    
    ########## iteration ##########
    for iteration, (images, labels) in enumerate(tqdm(train_loader)):
        # print('Iteration: {}'.format(iteration+1))
    
        images = images.to(device)
        labels = labels.to(device)

        # forward
        optimizer.zero_grad()
        outputs = net(images)

        loss = criterion(outputs, labels.long()) # outputs: float32, labels.long():int64

        # backward
        loss.backward()
        optimizer.step()
        
        # sum loss (all iterations)
        train_loss += loss.item()

        # max logit >> predictions
        # predictions = torch.max(outputs, axis=1).indices
        preds = torch.argmax(outputs, dim=1)

        # # sum acc (all iterations)
        # train_acc += torch.sum(preds == labels).item() / len(labels)

        # summary labels & predictions into list
        for label in labels:
            train_labels.append(label.item())
        for pred in preds:
            train_preds.append(pred.item())

    ########## training evaluation ##########
    # mean loss (1 epoch)
    epoch_train_loss = train_loss / len(train_loader)
    print('Train loss: {:.4f}'.format(epoch_train_loss), end='  ')
    # loss list (epochs)
    train_loss_list.append(epoch_train_loss)

    # # mean acc (1 epoch)
    # epoch_train_acc = train_acc / len(train_loader)
    # print('Train acc: {:.4f}'.format(epoch_train_acc), end='  ')
    # # acc list (epochs)
    # train_acc_list.append(epoch_train_acc)

    y_true = train_labels
    y_pred = train_preds

    # '''
    # mean acc (1 epoch)
    epoch_train_acc = accuracy_score(y_true, y_pred)
    print('Train acc: {:.4f}'.format(epoch_train_acc), end='  ')
    # acc list (epochs)
    train_acc_list.append(epoch_train_acc)

    # mean precision (1 epoch)
    epoch_train_precision = precision_score(y_true, y_pred, average='weighted') # average = [None, 'binary'(default), 'micro'(=acc), 'macro'(average of classes), 'samples', 'weighted'(weighted macro)]
    print('Train precision: {:.4f}'.format(epoch_train_precision), end='  ')
    # acc list (epochs)
    train_precision_list.append(epoch_train_precision)

    # mean recall (1 epoch)
    epoch_train_recall = recall_score(y_true, y_pred, average='weighted') # average = [None, 'binary'(default), 'micro'(=acc), 'macro'(average of classes), 'samples', 'weighted'(weighted macro)]
    print('Train recall: {:.4f}'.format(epoch_train_recall), end='  ')
    # recall list (epochs)
    train_recall_list.append(epoch_train_recall)

    # mean f1-score (1 epoch)
    epoch_train_f1_score = f1_score(y_true, y_pred, average='weighted') # average = [None, 'binary'(default), 'micro'(=acc), 'macro'(average of classes), 'samples', 'weighted'(weighted macro)]
    print('Train f1-score: {:.4f}'.format(epoch_train_f1_score), end='  ')
    # f1-score list (epochs)
    train_f1_score_list.append(epoch_train_f1_score)
    # '''

    ## train end time (1 epoch)
    train_end_time = time.time()
    print('time: {:.2f}s'.format(train_end_time - train_start_time))
    train_time += (train_end_time - train_start_time)


    ########## validation ##########
    ## val start time (1 epoch)
    val_start_time = time.time()
    print('-----Val mode-----')
    net.eval()
    
    with torch.no_grad():
        ########## iteration ##########
        for iteration, (images, labels) in enumerate(tqdm(val_loader)):
            # print('Iteration: {}'.format(iteration+1))
        
            images = images.to(device)
            labels = labels.to(device)
            
            # forward
            outputs = net(images)
            loss = criterion(outputs, labels.long())

            # sum loss (all iterations)
            val_loss += loss.item()
        
            # max logit >> predictions
            # predictions = torch.max(outputs, axis=1).indices
            preds = torch.argmax(outputs, dim=1)

            # # sum acc (all iterations)
            # val_acc += torch.sum(preds == labels).item() / len(labels)

            # summary labels & predictions into list
            for label in labels:
                val_labels.append(label.item())
            for pred in preds:
                val_preds.append(pred.item())


        ########## validation evaluation ##########
        # mean loss (1 epoch)
        epoch_val_loss = val_loss / len(val_loader)
        print('Val loss: {:.4f}'.format(epoch_val_loss), end='  ')
        # loss list (epochs)
        val_loss_list.append(epoch_val_loss)

        # # mean acc (1 epoch)
        # epoch_val_acc = val_acc / len(val_loader)
        # print('Val acc: {:.4f}'.format(epoch_val_acc), end='  ')
        # # acc list (epochs)
        # val_acc_list.append(epoch_val_acc)

        y_true = val_labels
        y_pred = val_preds

        # '''
        # mean acc (1 epoch)
        epoch_val_acc = accuracy_score(y_true, y_pred)
        print('Val acc: {:.4f}'.format(epoch_val_acc), end='  ')
        # acc list (epochs)
        val_acc_list.append(epoch_val_acc)

        # mean precision (1 epoch)
        epoch_val_precision = precision_score(y_true, y_pred, average='weighted') # average = [None, 'binary'(default), 'micro'(=acc), 'macro'(average of classes), 'samples', 'weighted'(weighted macro)]
        print('Val precision: {:.4f}'.format(epoch_val_precision), end='  ')
        # acc list (epochs)
        val_precision_list.append(epoch_val_precision)

        # mean recall (1 epoch)
        epoch_val_recall = recall_score(y_true, y_pred, average='weighted') # average = [None, 'binary'(default), 'micro'(=acc), 'macro'(average of classes), 'samples', 'weighted'(weighted macro)]
        print('Val recall: {:.4f}'.format(epoch_val_recall), end='  ')
        # recall list (epochs)
        val_recall_list.append(epoch_val_recall)

        # mean f1-score (1 epoch)
        epoch_val_f1_score = f1_score(y_true, y_pred, average='weighted') # average = [None, 'binary'(default), 'micro'(=acc), 'macro'(average of classes), 'samples', 'weighted'(weighted macro)]
        print('Val f1-score: {:.4f}'.format(epoch_val_f1_score), end='  ')
        # f1-score list (epochs)
        val_f1_score_list.append(epoch_val_f1_score)
        # '''

        ## val end time (1 epoch)
        val_end_time = time.time()
        print('time: {:.2f}s'.format(val_end_time - val_start_time))
        val_time += (val_end_time - val_start_time)




########## all time ##########
print('====================')
print('train all time: {:.2f}s'.format(train_time))
print('val all time: {:.2f}s'.format(val_time))
print('====================')


########## save weights ##########
weights_path = 'weights/' + NAME_NET + str(EPOCH) + '.pth'
torch.save(net.to(device).state_dict(), weights_path)
print(weights_path + ': saved!')
print('====================')


########## visualize loss ##########
plt.figure('loss')
plt.title('Train loss & Val loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
# train loss
plt.plot(range(1, EPOCH+1), train_loss_list, 'b-', label='Train loss')
# val loss
plt.plot(range(1, EPOCH+1), val_loss_list, 'r-', label='Val loss')
plt.legend()
plt.savefig('weights/' + NAME_NET + str(EPOCH) + '_loss' + '.png')
plt.show()


########## visualize acc ##########
plt.figure('accuracy')
plt.title('Train acc & Val acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
# train acc
plt.plot(range(1, EPOCH+1), train_acc_list, 'b-', label='Train acc')
# val acc
plt.plot(range(1, EPOCH+1), val_acc_list, 'r-', label='Val acc')
plt.legend()
plt.savefig('weights/' + NAME_NET + str(EPOCH) + '_acc' + '.png')
plt.show()


########## visualize precision ##########
plt.figure('precision')
plt.title('Train precision & Val precision')
plt.xlabel('Epoch')
plt.ylabel('Precision')
# train acc
plt.plot(range(1, EPOCH+1), train_precision_list, 'b-', label='Train precision')
# val acc
plt.plot(range(1, EPOCH+1), val_precision_list, 'r-', label='Val precision')
plt.legend()
plt.savefig('weights/' + NAME_NET + str(EPOCH) + '_precision' + '.png')
plt.show()


########## visualize recall ##########
plt.figure('recall')
plt.title('Train recall & Val recall')
plt.xlabel('Epoch')
plt.ylabel('Recall')
# train acc
plt.plot(range(1, EPOCH+1), train_recall_list, 'b-', label='Train recall')
# val acc
plt.plot(range(1, EPOCH+1), val_recall_list, 'r-', label='Val recall')
plt.legend()
plt.savefig('weights/' + NAME_NET + str(EPOCH) + '_recall' + '.png')
plt.show()


########## visualize f1-score ##########
plt.figure('f1-score')
plt.title('Train f1-score & Val f1-score')
plt.xlabel('Epoch')
plt.ylabel('F1-score')
# train acc
plt.plot(range(1, EPOCH+1), train_f1_score_list, 'b-', label='Train f1-score')
# val acc
plt.plot(range(1, EPOCH+1), val_f1_score_list, 'r-', label='Val f1-score')
plt.legend()
plt.savefig('weights/' + NAME_NET + str(EPOCH) + '_f1-score' + '.png')
plt.show()


########## visualize confusion matrix ##########
# visual_CM(y_true=y_true, y_pred=y_pred, name_classes=NAME_CLASSES, normalized=True) # 2 or multi-class
visual_CM_2class(y_true=y_true, y_pred=y_pred) # only 2
