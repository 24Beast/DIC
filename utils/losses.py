import torch

# Modified Functions

# BCE Loss
BCE_loss_instance = torch.nn.BCELoss()


def ModifiedBCELoss(y_pred, y):
    return 1 / BCE_loss_instance(y_pred, y)


# CrossEntropy Loss
CE_loss_instance = torch.nn.CrossEntropyLoss()


def ModifiedCELoss(y_pred, y):
    return 1 / CE_loss_instance(y_pred, y)
