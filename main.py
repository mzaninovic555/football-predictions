import datetime
from math import floor

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from football_dataset import FootballDataset

writer = SummaryWriter()

dataset = FootballDataset('./data/onehotz.csv')

input_features = dataset.__getitem__(0)[0].size(dim=0)
output_features = dataset.__getitem__(0)[1].size(dim=0)

batch_size = 32
epochs = 60
learning_rate = 0.003
dropout_rate = 0.25
hidden_layer_size = floor(input_features * 16)
hidden_layer_size2 = floor(input_features * 8)
hidden_layer_size3 = floor(input_features * 4)
hidden_layer_size4 = floor(input_features * 2)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

len_80 = int(len(dataset) * 0.8)
len_20 = len(dataset) - len_80
train, validation = torch.utils.data.random_split(dataset, [len_80, len_20])
train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(validation, batch_size=batch_size, shuffle=True)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.activation_function = nn.ReLU()
        # self.hidden_func = nn.LeakyReLU()
        # self.hidden_func = nn.Tanh()
        # self.hidden_func = nn.Sigmoid()

        self.dropout = nn.Dropout(dropout_rate)

        self.sequential = nn.Sequential(
            # input
            nn.Linear(input_features, hidden_layer_size),
            nn.BatchNorm1d(hidden_layer_size2),
            self.activation_function,

            # hidden 1
            nn.Linear(hidden_layer_size, hidden_layer_size2),
            nn.BatchNorm1d(hidden_layer_size2),
            self.activation_function,
            self.dropout,

            # hidden 2
            nn.Linear(hidden_layer_size2, hidden_layer_size3),
            nn.BatchNorm1d(hidden_layer_size3),
            self.activation_function,
            self.dropout,

            # hidden 3
            nn.Linear(hidden_layer_size3, hidden_layer_size4),
            nn.BatchNorm1d(hidden_layer_size4),
            self.activation_function,

            # output
            nn.Linear(hidden_layer_size4, output_features),
        )

    def forward(self, x):
        return self.sequential(x)


def train():
    model = NeuralNetwork().to(device)
    print(model)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    total_correct = 0
    total_total = 0
    for epoch in range(epochs):
        i = 0

        correct = 0
        total = 0
        for data, target in train_loader:
            data = data.to(device)
            target = target.to(device)

            # Clear gradients
            optimizer.zero_grad()

            # Get predictions
            output = model(data)

            # Calculate loss
            loss = criterion(output, target)
            writer.add_scalar("Loss/train", loss, epoch)

            # Backpropagate and update weights
            loss.backward()
            optimizer.step()

            predicted = torch.argmax(output, dim=1)
            predicted_one_hot = torch.nn.functional.one_hot(predicted,
                                                            num_classes=3)

            correct += (predicted_one_hot == target).all(dim=1).sum()
            total += target.size(0)
            accuracy = (correct / total) * 100

            total_correct += correct
            total_total += total

            writer.add_scalar("Accuracy", accuracy, epoch)

            if (i + 1) % 100 == 0:
                print(
                    'Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {}'.format(
                        epoch + 1, epochs, i, len(train_loader), loss.item(),
                        accuracy))
            i += 1

    print(
        f"Training finished, total accuracy: {(total_correct / total_total) * 100}")

    writer.flush()
    writer.close()

    now = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    torch.save(model.state_dict(), f'./models/football_predictor_{now}_dict.pt')
    torch.save(model, f'./models/football_predictor_{now}_model.pt')

    model.eval()

    i = 0
    valid_correct = 0
    valid_total = 0
    for data, target in validation_loader:
        data = data.to(device)
        target = target.to(device)

        # Get predictions
        output = model(data)

        # Calculate loss
        loss = criterion(output, target)

        predicted = torch.argmax(output, dim=1)
        predicted_one_hot = torch.nn.functional.one_hot(predicted,
                                                        num_classes=3)

        valid_correct += (predicted_one_hot == target).all(dim=1).sum()
        valid_total += target.size(0)
        accuracy = (valid_correct / valid_total) * 100

        if (i + 1) % 100 == 0:
            print(
                'Step [{}/{}], Loss: {:.4f}, Accuracy: {}'.format(
                    i, len(validation_loader), loss.item(),
                    accuracy))
        i += 1

    print(
        f"Validation finished, total accuracy: {(valid_correct / valid_total) * 100}")


if __name__ == '__main__':
    train()
