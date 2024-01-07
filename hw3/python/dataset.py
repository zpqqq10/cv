import os
import cv2
import glob
import numpy as np

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))

def makedirs(_path):
    path = os.path.join(CURRENT_DIR, _path)
    if not os.path.exists(path):
        os.makedirs(path)


# load one dataset
def load_dataset(path):
    dataset = []
    labelset = []
    labels = glob.glob(os.path.join(path, "*"))
    for label in labels:
        imgs = glob.glob(os.path.join(label, "*.jpg"))
        _label = os.path.basename(label).lower()
        for img_name in imgs:
            image = cv2.imread(os.path.join(label, img_name))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            dataset.append(image)
            labelset.append(_label)
            
    sorted_labels = [os.path.basename(label).lower() for label in labels]
    sorted_labels.sort()
    return dataset, labelset, sorted_labels

def load_data(path = 'data'): 
    print('loading train set...')
    data_train, label_train, labels = load_dataset(os.path.join(CURRENT_DIR, path, "train"))
    print('loading test set...')
    data_test, label_test, labels = load_dataset(os.path.join(CURRENT_DIR, path, "test"))
    return data_train, label_train, data_test, label_test, labels
        


if __name__ == "__main__":
    train, test = load_data()
    breakpoint()