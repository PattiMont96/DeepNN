"""
train.py

Training utilities for DeepNN models.
"""

import torch
import torch.nn as nn
import torch.optim as optim


def train_model(model, data, targets, epochs=100, learning_rate=0.001):
    """
    Train a neural network using gradient descent.

    Parameters
    ----------
    model : torch.nn.Module
        Neural network model.
    data : torch.Tensor
        Input training data.
    targets : torch.Tensor
        Ground truth labels or regression targets.
    epochs : int, optional
        Number of training iterations.
    learning_rate : float, optional
        Optimizer learning rate.

    Returns
    -------
    list
        Training loss history.
    """

    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    loss_history = []

    for epoch in range(epochs):

        predictions = model(data)

        loss = loss_function(predictions, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    return loss_history