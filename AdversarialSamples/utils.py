import torch as T


def get_cls_accuracy(score, label):
    total = label.size(0)
    _, pred = T.max(score, dim=1)
    correct = T.sum(pred == label)
    accuracy = correct.float() / total

    return accuracy
