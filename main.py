import datetime
from math import floor

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from football_dataset import FootballDataset

writer = SummaryWriter()

dataset = FootballDataset('./data/Result_4.csv')

input_features = dataset.__getitem__(0)[0].size(dim=0)
output_features = dataset.__getitem__(0)[1].size(dim=0)

batch_size = 16
epochs = 15
learning_rate = 0.001
dropout_rate = 0
hidden_layer_size = floor(input_features * 1.5)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.relu = nn.ReLU()
        self.relu = nn.LeakyReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

        self.dropout = nn.Dropout(dropout_rate)
        self.l1 = nn.Linear(in_features=input_features,
                            out_features=hidden_layer_size)
        self.l2 = nn.Linear(in_features=hidden_layer_size,
                            out_features=hidden_layer_size)
        self.l3 = nn.Linear(in_features=hidden_layer_size,
                            out_features=hidden_layer_size)
        self.output = nn.Linear(in_features=hidden_layer_size,
                                out_features=output_features)

    def forward(self, x):
        x = self.l1(x)
        # x = self.sigmoid(x)
        # x = self.tanh(x)
        x = self.relu(x)
        # x = self.dropout(x)
        x = self.l2(x)
#         x = self.sigmoid(x)
#         x = self.tanh(x)
        x = self.relu(x)
#         x = self.dropout(x)
#         x = self.l3(x)
#         x = self.sigmoid(x)
#         x = self.tanh(x)
#         x = self.relu(x)
        x = self.output(x)
        return x


def train():
    model = NeuralNetwork().to(device)
    print(model)

    criterion = nn.CrossEntropyLoss()
    # criterion = nn.MSELoss()
    # criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(model.parameters())
    # optimizer = torch.optim.Adamax(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.NAdam(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate)

    total_correct = 0
    total_total = 0
    for epoch in range(epochs):
        i = 0

        correct = 0
        total = 0
        for data, target in dataloader:
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

            writer.add_scalar("Accuracy", accuracy, epoch)

            total_correct += correct
            total_total += total

            if (i + 1) % 100 == 0:
                print(
                    'Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {}'.format(
                        epoch + 1, epochs, i, len(dataloader), loss.item(),
                        accuracy))
            i += 1

    print(f"Training finished, total accuracy: {(total_correct / total_total) * 100}")

    writer.flush()
    writer.close()

    now = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    torch.save(model.state_dict(), f'./models/football_predictor_{now}_dict.pt')
    torch.save(model, f'./models/football_predictor_{now}_model.pt')


if __name__ == '__main__':
    train()
