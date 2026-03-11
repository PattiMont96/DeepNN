"""
utils.py

Utility functions used throughout the DeepNN project.
"""

import torch


def normalize_data(data):
    """
    Normalize tensor to zero mean and unit variance.

    Parameters
    ----------
    data : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        Normalized tensor.
    """

    mean = data.mean()
    std = data.std()

    return (data - mean) / std


def compute_accuracy(predictions, labels):
    """
    Compute classification accuracy.

    Parameters
    ----------
    predictions : torch.Tensor
        Model predictions (logits or probabilities).
    labels : torch.Tensor
        Ground truth labels.

    Returns
    -------
    float
        Accuracy value.
    """

    predicted_classes = torch.argmax(predictions, dim=1)
    correct = (predicted_classes == labels).sum().item()

    accuracy = correct / len(labels)

    return accuracy