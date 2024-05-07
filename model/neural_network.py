from math import floor

from torch import nn

input_features = 20
output_features = 3
dropout_rate = 0.2
hidden_layer_size = floor(input_features * 2)
hidden_layer_size2 = floor(input_features * 2)
hidden_layer_size3 = floor(input_features * 2)
hidden_layer_size4 = floor(input_features * 2)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.activation_function = nn.ReLU()

        self.dropout = nn.Dropout(dropout_rate)

        self.sequential = nn.Sequential(
            # input
            nn.Linear(input_features, hidden_layer_size),
            self.activation_function,

            # hidden 1
            nn.Linear(hidden_layer_size, hidden_layer_size2),
            self.activation_function,
            self.dropout,

            # hidden 2
            nn.Linear(hidden_layer_size2, hidden_layer_size3),
            self.activation_function,
            self.dropout,

            # hidden 3
            nn.Linear(hidden_layer_size3, hidden_layer_size4),
            self.activation_function,

            # output
            nn.Linear(hidden_layer_size4, output_features),
        )

    def forward(self, x):
        return self.sequential(x)
