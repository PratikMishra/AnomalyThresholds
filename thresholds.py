# Calculate threshold using different thresholding approaches
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix

def thresholds(trainError, valError=None, valLabels=None, mode='MaxRE', whisker=1.7239):
    # trainError: numpy array of reconstruction errors calculated on training set N (for MaxRE, StdRE and RedMaxRE approaches) or I_t (for OptRE, OptROC and OptPR approaches)
    # valError: numpy array of reconstruction errors calculated on combined validation set (I_v + O_v)
    # valLabels: numpy array of labels for combined validation set (I_v + O_v). I_v samples labeled as 0 and O_v samples labeled as 1.
    # mode: thresholding approach as described in paper.
    # whisker: whisker value for IQR analysis used in RedMaxRE thresholding approach.
    # Returns:
    #   threshold value for the corresponding mode or thresholding approach
    if mode=='MaxRE':
        #Deriving threshold as max of training errors.
        return np.max(trainError)
    elif mode=='StdRE':
        #Setting threshold as 3 standard deviations away from the mean
        return np.mean(trainError) + 3 * np.std(trainError)
    elif mode=='RedMaxRE':
        #Using IQR analysis to remove outliers and setting threshold as max of remaining training errors.
        fig1,ax1 = plt.subplots()
        bp = ax1.boxplot(trainError ,whis = whisker)
        outliers = len(bp['fliers'][0].get_ydata())
        q1 = bp['boxes'][0].get_ydata()[0]
        q3 = bp['boxes'][0].get_ydata()[2]
        q2 = bp['medians'][0].get_ydata()[0]
        inliers = trainError[(trainError>=(q3-((q3-q1)*whisker))) & (trainError<=(q3+((q3-q1)*whisker)))]
        return max(inliers)
    elif mode=='OptRE':
        #Determine threshold by cycling through training errors on validation set
        list_TPR=[]; list_TNR=[]; list_gmean=[]
        for i in range(len(trainError)):
            val_pred = valError>trainError[i,0]
            cfm = confusion_matrix(valLabels, val_pred)
            TPR = cfm[1,1]/(cfm[1,1]+cfm[1,0])
            TNR = cfm[0,0]/(cfm[0,0] + cfm[0,1])
            gmean = np.sqrt(TPR * TNR)
            list_TPR.append(TPR); list_TNR.append(TNR); list_gmean.append(gmean)
        maxIdx = np.argmax(list_gmean)
        return trainError[maxIdx,0]
    elif mode=='OptROC':
        #Determining threshold from ROC curve on validation data
        fpr, tpr, thresholds_ROC = roc_curve(valLabels, valError, pos_label = 1)
        list_TPR=[]; list_TNR=[]; list_gmean=[]
        for i in range(len(thresholds_ROC)):
            val_pred = valError>thresholds_ROC[i]
            cfm = confusion_matrix(valLabels, val_pred)
            TPR = cfm[1,1]/(cfm[1,1]+cfm[1,0])
            TNR = cfm[0,0]/(cfm[0,0] + cfm[0,1])
            gmean = np.sqrt(TPR * TNR)
            list_TPR.append(TPR); list_TNR.append(TNR); list_gmean.append(gmean)
        maxIdx = np.argmax(list_gmean)
        return thresholds_ROC[maxIdx]
    elif mode=='OptPR':
        #Determining threshold from PR curve on validation data
        prec, rec, thresholds_PR = precision_recall_curve(valLabels, valError, pos_label = 1)
        list_TPR=[]; list_TNR=[]; list_gmean=[]
        for i in range(len(thresholds_PR)):
            val_pred = valError>thresholds_PR[i]
            cfm = confusion_matrix(valLabels, val_pred)
            TPR = cfm[1,1]/(cfm[1,1]+cfm[1,0])
            TNR = cfm[0,0]/(cfm[0,0] + cfm[0,1])
            gmean = np.sqrt(TPR * TNR)
            list_TPR.append(TPR); list_TNR.append(TNR); list_gmean.append(gmean)
        maxIdx = np.argmax(list_gmean)
        return thresholds_PR[maxIdx]
