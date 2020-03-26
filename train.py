import torch.nn as nn
import torch
from model import net
from nn import NN
from dataset import TemplateDataset
from data_handler import DataHandler
from parameters import optimizer, loss_function, lr_scheduler, metric, config

model = 0

def run():
    global model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create dataset
    template_dataset = TemplateDataset(config)
    training_loader, validation_loader, test_loader = template_dataset.get_loaders()

    # Create the neural network
    model = NN(net, optimizer, loss_function, lr_scheduler, metric, device, config).to(device)

    # Create the data handler
    data_handler = DataHandler(training_loader, validation_loader, test_loader)

    for epoch in range(config['epochs']):
        # Training
        model.train()
        for i, data in enumerate(training_loader, 0):
            x, y = data
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = model.backpropagate(y_hat, y)
            result = model.evaluate(y_hat, y)
            data_handler.train_loss.append(loss)
            data_handler.train_metric.append(result)

        with torch.no_grad():
            model.eval()
            # Validating
            if validation_loader is not None:
                for i, data in enumerate(validation_loader, 0):
                    x, y = data
                    x, y = x.to(device), y.to(device)
                    y_hat = model(x)
                    _, loss = model.calculate_loss(y_hat, y)
                    result = model.evaluate(y_hat, y)
                    data_handler.valid_loss.append(loss)
                    data_handler.valid_metric.append(result)

            # Testing
            if test_loader is not None:
                for i, data in enumerate(test_loader, 0):
                    x, y = data
                    x, y = x.to(device), y.to(device)
                    y_hat = model(x)
                    _, loss = model.calculate_loss(y_hat, y)
                    result = model.evaluate(y_hat, y)
                    data_handler.test_loss.append(loss)
                    data_handler.test_metric.append(result)

        model.lr_scheduler_step()
        data_handler.epoch_end(epoch, model.get_lr())
    data_handler.plot(loss=config['plot']['loss'], metric=config['plot']['metric'])

if __name__ == '__main__':
    # torch.multiprocessing.freeze_support()
    try:
        run()
    except KeyboardInterrupt:
        model.save(config)