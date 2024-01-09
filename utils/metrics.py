import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, mean_squared_log_error, mean_absolute_error, r2_score
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.metrics import  f1_score, fbeta_score, matthews_corrcoef
from sklearn.metrics import log_loss, roc_auc_score




########## regression ##########
y_true = [1.0, 1.5, 2.0, 1.2, 1.8]
y_pred = [0.8, 1.5, 1.8, 1.3, 3.0]

##### MSE (Mean Squared Error)
mse = mean_squared_error(y_true, y_pred)

##### RMSE (Root Mean Squared Error)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))

##### RMSLE (Root Mean Squared Logarithmic Error)
rmsle = np.sqrt(mean_squared_log_error(y_true, y_pred))

##### MAE (Mean Absolute Error)
mae = mean_absolute_error(y_true, y_pred)

##### R^2 (Coefficient of Determination / R Squared) maximize R^2 = minimize RMSE
r2 = r2_score(y_true, y_pred)




########## 2-class classification (label-based) ##########
y_true = [0, 0, 1, 1, 0, 1, 1, 0]
y_pred = [1, 0, 1, 1, 0, 1, 1, 0]
labels = ['pole', 'person']

##### confusion matrix
cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel() # only 2-class
# normalized
cm_normalized = confusion_matrix(y_true, y_pred, normalize='true')
cm_normalized = np.around(cm_normalized, 2)
# visual confusion matrix
def visual_CM(cm, labels):
    plt.figure('confusion matrix')
    plt.xticks(range(len(cm)), labels)
    plt.yticks(range(len(cm)), labels)
    plt.title('confusion matrix')
    plt.xlabel('predicted')
    plt.ylabel('true')
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.colorbar()
    # write number
    for i in range(len(cm)):
        for j in range(len(cm[i])):
            plt.text(i-0.1, j, cm[j][i])
    plt.show()

##### accurcy
acc = accuracy_score(y_true, y_pred)

##### error rate
error_rate = 1-accuracy_score(y_true, y_pred)

##### precision
precision = precision_score(y_true, y_pred)

##### recall
recall = recall_score(y_true, y_pred)

##### F1-score
f1 = f1_score(y_true, y_pred)

##### Fbeta-score
fbeta = fbeta_score(y_true, y_pred, beta=2)

##### MCC (Matthews Correlation Coefficient)
mcc = matthews_corrcoef(y_true, y_pred)




########## 2-class classification (probability-based) ##########
##### logloss = cross-entropy = logistic loss (only 2-class)
y_true = [1, 0, 1, 1, 0, 1]
y_prob = [0.1, 0.2, 0.8, 0.8, 0.1, 0.3]
y_pred = [0, 0, 1, 1, 0, 0]

logloss = log_loss(y_true, y_prob)

##### ROC (Receiver Operation Characteristic Curve)



##### AUC (Area Under the ROC Curve)
auc = roc_auc_score(y_true, y_pred)


########## multi-class classification ##########
y_true = [0, 1, 2, 0, 1, 2, 0, 1]
y_pred = [0, 1, 2, 0, 1, 0, 1, 0]
labels = ['pole', 'person', 'vehicle']

##### confusion matrix
cm = confusion_matrix(y_true, y_pred)
# normalized
cm_normalized = confusion_matrix(y_true, y_pred, normalize='true')
cm_normalized = np.around(cm_normalized, 2)
# visual confusion matrix
def visual_CM(cm, labels):
    plt.figure('confusion matrix')
    plt.xticks(range(len(cm)), labels)
    plt.yticks(range(len(cm)), labels)
    plt.title('confusion matrix')
    plt.xlabel('predicted')
    plt.ylabel('true')
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.colorbar()
    # write number
    for i in range(len(cm)):
        for j in range(len(cm[i])):
            plt.text(i-0.1, j, cm[j][i])
    plt.show()

visual_CM(cm, labels)