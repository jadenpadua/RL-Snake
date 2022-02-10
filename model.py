import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

"""
@description: Feed forward Neural Network with Input, Hidden, Output layer
    - linear1: input_size as input and hidden size as output
    - linear2: hidden_size as input and output_size as output
"""


class Linear_QNet(nn.Module):
    """
    @description: init nn with 2 layers, hidden & output layer
    """

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    """
    @description: Applies activation function on linear layer1
        - Then feeds output of linear1 into linear2
    """

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    """
    @description: Saves our model.pth into dir
    """

    def save(self, file_name='model.pth'):
        path = './model'
        if not os.path.exists(path):
            os.makedirs(path)

        file_name = os.path.join(path, file_name)
        torch.save(self.state_dict(), file_name)


"""
@description: Trainer for actually training our model
"""


class QTrainer:
    """
    @description: Init our trainer with an optimizer and means squared error criterion
    """

    def __init__(self, model, lr, gamma):
        self.model = model
        self.lr = lr
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
