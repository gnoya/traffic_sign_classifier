import torch.nn as nn
import torch
from sklearn import metrics

# Edit your parameters
optimizer = torch.optim.Adam
loss_function = nn.CrossEntropyLoss
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR
metric = metrics.f1_score

config = {
    'epochs': 250,
    'learning_rate': 0.0008,
    'lr_scheduler': {
        'milestones': [80, 130, 170, 190],
        'gamma': 0.8
    },
    'dataset': {
        # If the dataset is in just one file
        'whole_set': None,
        'train_set_len': None,
        'valid_set_len': None,
        'test_set_len': None,
        # If the dataset is in multiple files
        'train_set': './dataset/GTSRB/Final_Training/Images/',
        'valid_set': './dataset/GTSRB/Final_Test/Images/',
        'test_set': None
    },
    'data_loader': {
        'batch_size': 3921,
        'shuffle': True,
        'num_workers': 0
    },
    'weight_decay': 0.0015,
    'plot': {
        'loss': True,
        'metric': True
    },
    'save_path': './checkpoint.pth.tar'
}