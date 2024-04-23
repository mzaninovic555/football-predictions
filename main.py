import torch
from torch import nn
from torch.utils.data import DataLoader

from football_dataset import FootballDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

dataset = FootballDataset('./data/Result_4.csv')

input_features = dataset.__getitem__(0)[0].size(dim=0)
output_features = dataset.__getitem__(0)[1].size(dim=0)

batch_size = 64
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

epochs = 50


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(in_features=input_features,
                      out_features=input_features * 3),
            nn.ReLU(),
            nn.Linear(in_features=input_features * 3,
                      out_features=input_features * 3),
            nn.ReLU(),
            nn.Linear(in_features=input_features * 3,
                      out_features=output_features),
            nn.Sigmoid()
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork().to(device)
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(epochs):
    i = 0

    total_correct = 0
    total_samples = 0

    for features, target in dataloader:
        features = features.to(device)
        target = target.to(device)

        # Clear gradients
        optimizer.zero_grad()

        # Get predictions
        output = model(features)

        # Calculate loss
        loss = criterion(output, target)

        # Backpropagate and update weights
        loss.backward()
        optimizer.step()

        total_correct += (output == target).sum().item()
        total_samples += target.size(0)

        # Print training progress (optional)
        if i % 100 == 0:
            accuracy = 100 * total_correct / total_samples
            print(
                'Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {}'.format(
                    epoch + 1, epochs, i, len(dataloader), loss.item(),
                    accuracy))
        i += 1

torch.save(model.state_dict(), './models/football_predictor.pt')

# current_position_distance
# home_goals_scored
# home_goals_conceded
# home_goal_difference
# away_goals_scored
# away_goals_conceded
# away_goal_difference
# home_last_5_wins
# home_last_5_draws
# home_last_5_losses
# away_last_5_wins
# away_last_5_draws
# away_last_5_losses
# home_season_wins
# home_season_draws
# home_season_losses
# away_season_wins
# away_season_draws
# away_season_losses
# home_win
# draw
# away_win

data = torch.tensor([1, 63, 37, 24, 77, 17, 60, 3, 1, 1, 3, 2, 0, 21, 6, 6, 27, 5, 1], dtype=torch.float).to(device)
output = model(data)
prediction = torch.argmax(output)
print(f"Prediction: {output}")
