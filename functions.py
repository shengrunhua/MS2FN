import torch
import torch.nn.functional as F
from sklearn.metrics import cohen_kappa_score
import numpy as np

def loss_fn(y_pred, y_true, classes, device):
    y_true = y_true.long()
    y_true = torch.eye(classes).to(device)[y_true]
    cross_entropy_loss = F.cross_entropy(y_pred, y_true)
    
    return cross_entropy_loss

def accuracy_fn(y_pred, y_true):
    predicted = torch.argmax(y_pred, 1)
    correct = (predicted == y_true).sum().item()
    total = y_true.size(0)
    
    return correct / total

def AA_fn(y_pred, y_true):
    class_number = y_pred.size(1)
    correct = np.zeros(class_number)
    total = np.zeros(class_number)
    y_pred = torch.argmax(y_pred, axis=-1)
    for i in range(class_number):
        total[i] = (y_true == i).sum().item()
        correct[i] = ((y_pred == y_true) * (y_true == i)).sum().item()

    return correct, total

def kappa_fn(y_pred, y_true):
    y_pred = np.argmax(y_pred, axis = -1)
    kappa = cohen_kappa_score(y_pred, y_true)
    
    return kappa