import os
import cv2
import numpy as np


def load_data(path):
    g_files = []
    ng_files = []
    X, y = [], []

    # get file paths for all good images
    g_dir = os.path.join(path,'train','good')
    for filename in os.listdir(g_dir):
        f = os.path.join(g_dir, filename)
        g_files.append(f)

    # load good images
    for file in g_files:
        X.append(cv2.imread(file))
        y.append(0)

    # get file paths for all not-good images
    ng_dir = os.path.join(path,'train','not-good')
    for filename in os.listdir(ng_dir):
        f = os.path.join(ng_dir, filename)
        ng_files.append(f)

    # load not-good images
    for file in ng_files:
        X.append(cv2.imread(file))
        y.append(1)

    c = list(zip(X, y))
    np.random.shuffle(c)
    X, y = zip(*c)

    return np.array(X), np.array(y)
