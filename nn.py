import torch
import torch.nn as nn
from sklearn import metrics
import numpy as np
import torch.nn.functional as F

class NN(nn.Module):
    def __init__(self, model, optimizer, loss_function, lr_scheduler, metric, device, config):
        super(NN, self).__init__()
        # Set up the model
        self.model = model

        # Set up the optimizer
        self.optimizer = optimizer(self.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])

        # Set up the loss function
        self.loss_function = loss_function()

        # Set up the learning rate scheduler
        self.lr_scheduler = lr_scheduler(self.optimizer, config['lr_scheduler']['milestones'], config['lr_scheduler']['gamma']) if lr_scheduler is not None else None

        # Set up the metric to evaluate
        self.metric = metric

        # Save the model device
        self.device = device

    def forward(self, x):
        return self.model(x)

    def backpropagate(self, y_pred, y):
        loss, _ = self.calculate_loss(y_pred, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def calculate_loss(self, y_pred, y):
        loss = self.loss_function(y_pred, y)
        return loss, loss.item()

    def lr_scheduler_step(self):
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            lr = param_group['lr']
        return lr

    def evaluate(self, y_pred, y):
        # TODO: optimize this function
        if str(self.device) == 'cuda':
            y_pred = y_pred.cpu()
            y = y.cpu()
        average = None
        if str(self.loss_function) == 'CrossEntropyLoss()':
            y_pred = F.softmax(y_pred)
            average = 'weighted'
            values, indices = y_pred.max(1)
            y_pred = indices.detach().numpy()
        elif str(self.loss_function) == 'BCELoss()':
            y_pred = y_pred.detach().numpy()
            y_pred = (y_pred > 0.5) * 1
        else:
            return 0

        y = y.detach().numpy()
        f1 = self.metric(y, y_pred, average=average, labels=np.unique(y_pred))

        return f1

    def save(self, config):
        data = {
            'optimizer': self.optimizer.state_dict(),
            'weights': self.state_dict()
        }
        torch.save(data, config['save_path'])