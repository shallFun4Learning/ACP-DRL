import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, accuracy_score, auc, matthews_corrcoef, confusion_matrix,average_precision_score
import torch
from torch import nn, scalar_tensor

# def getMetrics(y_true, y_pred, y_proba):


def getMetrics(eval_preds):
    logits, y_true = eval_preds
    sm = nn.Softmax(dim=1)
    y_proba = sm(torch.tensor(logits))
    y_proba = np.array(y_proba)
    # if NUM_LABELS == 1:
    # logits01 = np.where(logits < 0.5, 0, 1)
    # logits01 = np.array(y_proba)
    logits01 = np.argmax(y_proba, axis=-1).flatten()
    y_pred = logits01.astype(np.int16)
    # labels = labels.astype(np.int16)
    # y_pred = labels.flatten()

    ACC = accuracy_score(y_true, y_pred)
    MCC = matthews_corrcoef(y_true, y_pred)
    CM = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = CM.ravel()
    Sn = tp/(tp+fn)
    Sp = tn/(tn+fp)
    FPR, TPR, thresholds_ = roc_curve(y_true, y_proba[:, 1])
    AUC = auc(FPR, TPR)
    AUPR=average_precision_score(y_true, y_proba[:, 1])
    return {'ACC': ACC, 'MCC': MCC, 'Sn': Sn, 'Sp': Sp, 'AUC': AUC,'AUPR':AUPR}


def getScore(logits):
    sm = nn.Softmax(dim=1)
    y_proba = sm(torch.tensor(logits))
    y_proba = np.array(y_proba)
    return y_proba


def getPredictLabel(logits):
    sm = nn.Softmax(dim=1)
    y_proba = sm(torch.tensor(logits))
    y_proba = np.array(y_proba)
    logits01 = np.argmax(y_proba, axis=-1).flatten()
    y_pred = logits01.astype(np.int16)
    return y_pred
