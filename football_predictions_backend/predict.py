import os

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = torch.jit.load('./model/prediction_model.pt', map_location=device)
model.eval()


def predict(prediction_request):
    data = torch.tensor([[
        prediction_request['home_position'],
        prediction_request['away_position'],
        prediction_request['home_goals_scored'],
        prediction_request['home_goals_conceded'],
        prediction_request['home_goal_difference'],
        prediction_request['away_goals_scored'],
        prediction_request['away_goals_conceded'],
        prediction_request['away_goal_difference'],
        prediction_request['home_last_5_wins'],
        prediction_request['home_last_5_draws'],
        prediction_request['home_last_5_losses'],
        prediction_request['away_last_5_wins'],
        prediction_request['away_last_5_draws'],
        prediction_request['away_last_5_losses'],
        prediction_request['home_season_wins'],
        prediction_request['home_season_draws'],
        prediction_request['home_season_losses'],
        prediction_request['away_season_wins'],
        prediction_request['away_season_draws'],
        prediction_request['away_season_losses']
    ]], dtype=torch.float32).to(device)
    output = model(data)
    print(f"Prediction output: {output}")
    predicted = torch.nn.functional.softmax(output, dim=1).detach().cpu().numpy()
    print(f"Prediction output: {predicted}")
    return predicted
