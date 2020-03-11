import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import csv
import numpy as np
import cv2
from utils import * 


epoch_loss = []
epoch_train_acc = []
epoch_test_acc = []

# labels: numero entre 0 y 42
# y debe tener dimensiones (m, 43)
def run(number, image_size, model, learning_rate, epochs, milestones, gamma):
    global epoch_loss, epoch_train_acc, epoch_test_acc
    torch.multiprocessing.freeze_support()

    print()
    print('*************************************')
    print('TRY: ', number)
    print(image_size, learning_rate, epochs, milestones, gamma)
    print('*************************************')
    print()

    classes = 43

    params = {'batch_size': 3921,
            'shuffle': True,
            'num_workers': 0}

    # Create Dataset
    training_set = TrafficSignDataset('./GTSRB/Final_Training/Images/', classes, image_size, False)
    training_loader = DataLoader(dataset=training_set, **params)

    counting_labels = np.zeros((classes))
    for label in training_set.labels:
        counting_labels[int(label.item())] += 1

    class_weights = 1 / counting_labels
    class_weights = class_weights / class_weights.sum()

    class_weights = torch.from_numpy(class_weights).float()
    class_weights = class_weights.to('cuda')

    validation_set = TrafficSignDataset('./GTSRB/Final_Test/Images/', classes, image_size, True)
    validation_loader = DataLoader(dataset=validation_set, **params)

    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    # loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0015)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    for t in range(epochs):
        model.train()
        train_f1 = []
        train_acc = []
        total_loss = []
        for i, data in enumerate(training_loader, 0):
            local_x, local_y = data
            local_x = local_x.to('cuda')
            local_y = local_y.to('cuda')

            y_pred = model(local_x)
            loss = loss_fn(y_pred, local_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tp, fp, tn, fn, acc, prec, recall, f1 = evaluate_model(y_pred, local_y)

            train_f1.append(f1)
            train_acc.append(acc)
            total_loss.append(loss.item())

        train_f1 = np.asarray(train_f1)
        train_f1 = np.mean(train_f1)

        train_acc = np.asarray(train_acc)
        train_acc = np.mean(train_acc)

        total_loss = np.asarray(total_loss)
        total_loss = np.mean(total_loss)

        scheduler.step()
        with torch.no_grad():
            model.eval()
            test_f1 = []
            test_acc = []
            for i, data in enumerate(validation_loader, 0):
                local_x, local_y = data
                local_x = local_x.to('cuda')
                local_y = local_y.to('cuda')

                y_pred_test = model(local_x)
                tp, fp, tn, fn, acc, prec, recall, f1 = evaluate_model(y_pred_test, local_y)
                test_f1.append(f1)
                test_acc.append(acc)

            test_f1 = np.asarray(test_f1)
            test_f1 = np.mean(test_f1)
            test_acc = np.asarray(test_acc)
            test_acc = np.mean(test_acc)
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
        
        epoch_loss.append(total_loss)
        epoch_train_acc.append(train_f1)
        epoch_test_acc.append(test_f1)
        print('Epoch {0}: Cost: {1:.4f} | Training F1: {2:.4f} | Testing F1: {3:.4f} | Learning rate: {4:.6f}'.format(t, total_loss, train_f1, test_f1, lr))
        # print('Epoch {0}: Cost: {1:.4f} | Training F1: {2:.4f} | Training Acc: {3:.4f} | Learning rate: {4:.4f}'.format(t, total_loss, train_f1, train_acc, lr))


if __name__ == '__main__':
    try:
        image_size = 40
        classes = 43

        model = nn.Sequential(
            nn.Linear(image_size*image_size, 2*4096),
            nn.Dropout(0.35),
            nn.BatchNorm1d(num_features=2*4096),
            nn.ReLU(),
            nn.Linear(2*4096, 4096),
            nn.Dropout(0.50),
            nn.BatchNorm1d(num_features=4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.Dropout(0.50),
            nn.BatchNorm1d(num_features=4096),
            nn.ReLU(),
            nn.Linear(4096, classes)
        )
        model = model.to('cuda')

        learning_rate = 0.0008
        epochs = 250
        milestones = [80, 130, 170, 190]
        gamma = 0.8
        run(1, image_size, model, learning_rate, epochs, milestones, gamma)
        # run(2, image_size, model, 0.0001, epochs, milestones, gamma)
        # run(2, image_size, model, 0.005, epochs, [75, 125], 0.33)
    except KeyboardInterrupt:
        plot_data(epoch_loss, epoch_train_acc, epoch_test_acc)
