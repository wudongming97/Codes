import torch as T
import numpy as np


def get_cls_accuracy(score, label):
    total = label.size(0)
    _, pred = T.max(score, dim=1)
    correct = T.sum(pred == label)
    accuracy = correct.float() / total

    return accuracy


def to_categorical(labels):
    one_hot = np.zeros((labels.shape[0], labels.max() + 1))
    one_hot[np.arange(labels.shape[0]), labels] = 1
    return one_hot
