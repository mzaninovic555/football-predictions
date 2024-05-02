import torch

from main import NeuralNetwork

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load(
    "./models/football_predictor_2024_04_27_18_57_29_model.pt").to(device)
model.eval()

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

# milan - inter
# data = torch.tensor(
#     [1, 63, 37, 24, 77, 17, 60, 3, 1, 1, 3, 2, 0, 21, 6, 6, 27, 5, 1],
#     dtype=torch.float).to(device)

# milan - juve
# data = torch.tensor(
#     [1, 47, 26, 21, 64, 39, 25, 1, 3, 1, 3, 1, 1, 18, 10, 5, 21, 6, 6],
#     dtype=torch.float).to(device)

data = torch.tensor(
    [1, 62, 20, 42, 58, 24, 34, 5, 0, 0, 5, 0, 0, 22, 5, 4, 22, 6, 4],
    dtype=torch.float).to(device)

output = model(data)
print()
print(f"Prediction (home_win, draw, away_win): {output}")
print(
    f"Prediction (home_win, draw, away_win): {torch.nn.functional.softmax(output, dim=0)}")

predicted = torch.argmax(output, dim=0)
predicted_one_hot = torch.nn.functional.one_hot(predicted,
                                                num_classes=3)
print(f"Prediction one_hot (home_win, draw, away_win): {predicted_one_hot}")

