import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import csv
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn import metrics
from sklearn.metrics import accuracy_score

class TrafficSignDataset(Dataset):
    def __init__(self, rootpath, classes, image_size, test=False):
        if test:
            images, labels = readTrafficSigns_test(rootpath, classes, image_size)
        else:
            images, labels = readTrafficSigns(rootpath, classes, image_size)
        
        x = np.asarray(images) / 255
        y = np.asarray(labels).astype(float)

        self.length = y.shape[0]

        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).long()

        self.samples = x
        self.labels = y

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        x = self.samples[index]
        y = self.labels[index]

        return x, y

# function for reading the images
# arguments: path to the traffic sign data, for example './GTSRB/Training'
# returns: list of images, list of corresponding labels 
def readTrafficSigns(rootpath, classes, image_size):
    '''Reads traffic sign data for German Traffic Sign Recognition Benchmark.

    Arguments: path to the traffic sign data, for example './GTSRB/Training'
    Returns:   list of images, list of corresponding labels'''
    images = [] # images
    labels = [] # corresponding labels
    # loop over all 43 classes
    total_counter = 0
    for c in range(0, classes):
        prefix = rootpath + '/' + format(c, '05d') + '/' # subdirectory for class
        gtFile = open(prefix + 'GT-'+ format(c, '05d') + '.csv') # annotations file
        gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
        next(gtReader) # skip header
        # loop over all images in current annotations file
        counter = 0
        for row in gtReader:
            # if int(row[1]) >= image_size and int(row[2]) >= image_size:
            if True:
                img = cv2.imread(prefix + row[0]) # the 1th column is the filename
                grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                resized_image = cv2.resize(grayImage, (image_size, image_size))
                np_image = np.asarray(resized_image).ravel()
                images.append(np_image) 
                labels.append(row[7]) # the 8th column is the label
                counter += 1
                total_counter += 1
        print('Label {} has {} images'.format(c, counter))
        gtFile.close()
    print('Training set has {} images'.format(total_counter))
    return images, labels


# function for reading the images
# arguments: path to the traffic sign data, for example './GTSRB/Training'
# returns: list of images, list of corresponding labels 
def readTrafficSigns_test(rootpath, classes, image_size):
    '''Reads traffic sign data for German Traffic Sign Recognition Benchmark.

    Arguments: path to the traffic sign data, for example './GTSRB/Training'
    Returns:   list of images, list of corresponding labels'''
    images = [] # images
    labels = [] # corresponding labels

    gtFile = open(rootpath + 'GT-final_test.csv') # annotations file
    gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
    next(gtReader) # skip header
    # loop over all images in current annotations file
    counter = 0
    for row in gtReader:
        if int(row[7]) < classes:
            # if int(row[1]) >= image_size and int(row[2]) >= image_size:
            if True:
                img = cv2.imread(rootpath + row[0]) # the 1th column is the filename
                grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                resized_image = cv2.resize(grayImage, (image_size, image_size))
                np_image = np.asarray(resized_image).ravel()
                images.append(np_image) 
                labels.append(row[7]) # the 8th column is the label
                counter += 1
    print('Test set has {} images'.format(counter))
    gtFile.close()
    return images, labels

def evaluate_acc(y_pred, y, test=False):
    y = y.detach().numpy()
    y_pred = F.softmax(y_pred)

    lesser_than_threshold = y_pred <= 0.5
    y_pred[lesser_than_threshold] = 0
    values, indices = y_pred.max(1)
    indices = indices.detach().numpy()

    matches = (indices == y)
    tp = matches.sum()
    acc = tp / len(matches)

    return acc

def plot_data(losses, train_accs, validation_accs):
    #------------------------------------- Loss plot -------------------------------------- #
    plt.plot(losses)
    plt.title('Loss plot')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.show()

    #----------------------------------- Train acc plot -------------------------------------- #
    plt.plot(train_accs)
    plt.title('Train acc plot')
    plt.ylabel('Train accuracy')
    plt.xlabel('Epochs')
    plt.show()

    #--------------------------------- Validation acc plot ------------------------------------ #
    plt.plot(validation_accs)
    plt.title('Validation acc plot')
    plt.ylabel('Validation accuracy')
    plt.xlabel('Epochs')
    plt.show()

def evaluate_model(y_pred, y):
    y = y.cpu().detach().numpy()
    y_pred = F.softmax(y_pred)

    values, indices = y_pred.max(1)
    indices = indices.cpu().detach().numpy()

    f1 = f1_score(y, indices, average='weighted')
    acc = accuracy_score(y, indices)

    return 0, 0, 0, 0, acc, 0, 0, f1