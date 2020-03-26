import torch.nn as nn

# Edit your model
net = nn.Sequential(
    nn.Linear(40*40, 2*4096),
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
    nn.Linear(4096, 43)
)