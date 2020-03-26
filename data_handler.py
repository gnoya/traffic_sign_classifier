import numpy as np
import matplotlib.pyplot as plt

# TODO: multiple metrics support
class DataHandler():
    def __init__(self, train_loader, valid_loader, test_loader):
        self.train = train_loader is not None
        self.valid = valid_loader is not None
        self.test = test_loader is not None

        self.train_loss = []
        self.valid_loss = []
        self.test_loss = []

        self.train_metric = []
        self.valid_metric = []
        self.test_metric = []

        self.total_train_loss = np.array([])
        self.total_valid_loss = np.array([])
        self.total_test_loss = np.array([])

        self.total_train_metric = np.array([])
        self.total_valid_metric = np.array([])
        self.total_test_metric = np.array([])

        self.i = 0

    def epoch_end(self, epoch, lr):
        self.mean_data()
        self.reset_data()

        print('\nEpoch {0} | Learning rate: {1:.8f}'.format(epoch, lr))
        if self.train:
            print('Training   | Cost: {0:.4f} | Metric: {1:.4f}'.format(self.total_train_loss[-1], self.total_train_metric[-1]))
        if self.valid:
            print('Validation | Cost: {0:.4f} | Metric: {1:.4f}'.format(self.total_valid_loss[-1], self.total_valid_metric[-1]))
        if self.test:
            print('Test       | Cost: {0:.4f} | Metric: {1:.4f}'.format(self.total_test_loss[-1], self.total_test_metric[-1]))
        
    def mean_data(self):
        if self.train:
            train_loss = np.mean(self.train_loss)
            train_metric = np.mean(self.train_metric)
            self.total_train_loss = np.append(self.total_train_loss, train_loss)
            self.total_train_metric = np.append(self.total_train_metric, train_metric)

        if self.valid:
            valid_loss = np.mean(self.valid_loss)
            valid_metric = np.mean(self.valid_metric)
            self.total_valid_loss = np.append(self.total_valid_loss, valid_loss)
            self.total_valid_metric = np.append(self.total_valid_metric, valid_metric)

        if self.test:
            test_loss = np.mean(self.test_loss)
            test_metric = np.mean(self.test_metric)
            self.total_test_loss = np.append(self.total_test_loss, test_loss)
            self.total_test_metric = np.append(self.total_test_metric, test_metric)  
        
    def plot(self, loss, metric):
        if loss:
            self.plot_loss(False)
        if metric:
            self.plot_metric(False)
        self.i = 0
        plt.show()

    def plot_loss(self, show):
        if self.train:
            self.figure(self.total_train_loss, 'Training loss', 'Epochs', 'Train loss')

        if self.valid:
            self.figure(self.total_valid_loss, 'Validation loss', 'Epochs', 'Valid loss')

        if self.test:
            self.figure(self.total_test_loss, 'Testing loss', 'Epochs', 'Test loss')
        
        if show:
            plt.show()

    def plot_metric(self, show):
        if self.train:
            self.figure(self.total_train_metric, 'Training metric', 'Epochs', 'Train metric')

        if self.valid:
            self.figure(self.total_valid_metric, 'Validation metric', 'Epochs', 'Valid metric')

        if self.test:
            self.figure(self.total_test_metric, 'Testing metric', 'Epochs', 'Test metric')
        
        if show:
            plt.show()

    def figure(self, data, title, xlabel, ylabel):
        self.i += 1
        plt.figure(self.i)
        plt.plot(data)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        
    def reset_data(self):
        self.reset_losses()
        self.reset_metrics()

    def reset_losses(self):
        self.train_loss = []
        self.valid_loss = []
        self.test_loss = []

    def reset_metrics(self):
        self.train_metric = []
        self.valid_metric = []
        self.test_metric = []