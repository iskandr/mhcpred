import numpy as np


def find_best_threshold_accuracy(x, y):
    thresholds = np.unique(x)
    best_accuracy = 0
    best_threshold = None
    for t in thresholds:
        mask = (x<=t)
        acc_below = np.mean(mask == y)
        acc_above = np.mean(~mask == y)
        accuracy = max(acc_below, acc_above)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = t
    return best_accuracy, best_threshold

def find_threshold_pairs(x1, x2, y):
    thresholds1 = np.unique(x1)
    thresholds2 = np.unique(x2)
    best_accuracy = 0
    best_threshold = None
    for t1 in thresholds1:
        mask1 = (x1<=t1)
        for t2 in thresholds2:
            mask2 = (x2<=t2)
            accuracy = max(
                np.mean(mask == y)
                for mask in
                [
                  mask1 & mask2,
                  mask1 & ~mask2,
                  ~mask1 & mask2,
                  ~mask1 & ~mask2
                ]
            )
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = (t1, t2)
    return best_accuracy, best_threshold