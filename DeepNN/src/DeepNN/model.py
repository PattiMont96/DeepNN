"""
model.py

Neural network architectures used in the DeepNN project.
"""

import torch
import torch.nn as nn


class SimpleNN(nn.Module):
    """
    Fully connected neural network.

    This model consists of three linear layers with ReLU activation.

    Parameters
    ----------
    input_size : int
        Number of input features.
    hidden1_size : int
        Number of neurons in the first hidden layer.
    hidden2_size : int
        Number of neurons in the second hidden layer.
    output_size : int
        Number of output neurons.
    """

    def __init__(self, input_size=10, hidden1_size=20, hidden2_size=10, output_size=1):
        super().__init__()

        self.layer1 = nn.Linear(input_size, hidden1_size)
        self.layer2 = nn.Linear(hidden1_size, hidden2_size)
        self.layer3 = nn.Linear(hidden2_size, output_size)

    def forward(self, x):
        """
        Perform forward propagation.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape (batch_size, input_size).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, output_size).
        """

        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)

        return x