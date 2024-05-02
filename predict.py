import os

import torch

from model.neural_network import NeuralNetwork

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = torch.jit.load(f'{os.getcwd()}/models/football_predictor_2024_05_02_19_47_34_dict.pt')
model.eval()


def predict(params):
    output = model(params)
    predicted = torch.argmax(output, dim=0)
    return predicted
