# Importing Libraries
import copy
import math
import torch
import numpy as np
import torch.optim as optim
from typing import Callable, Union, Literal
from DLA import DLA


if __name__ == "__main__":
    # Test case
    import pandas as pd
    from attackerModels.ANN import simpleDenseModel

    # Data Initialization
    from utils.datacreator import StabilityExp

    NUM_SAMPLES = 16384
    ATTACKER_WIDTHS = [i for i in range(1, 16)]
    TRAIN_PARAMS = {
        "learning_rate": 0.05,
        "loss_function": "mse",
        "epochs": 100,
        "batch_size": 64,
    }

    P, D, M = StabilityExp(NUM_SAMPLES)
    P = torch.tensor(P, dtype=torch.float).reshape(-1, 1)
    D = torch.tensor(D, dtype=torch.float).reshape(-1, 1)
    M = torch.tensor(M, dtype=torch.float).reshape(-1, 1)

    # Calculating Params
    model_mse = RMSE(M, D)

    # Parameter Initialization

    for num, width in enumerate(ATTACKER_WIDTHS):
        # Attacker Model Initialization
        attackerModel = simpleDenseModel(
            1, width, 1, numFirst=1, activations=["relu", "relu", "relu"]
        )

        # Parameter Initialization
        dla = DLA(
            {"attacker_AtoT": attackerModel, "attacker_TtoA": attackerModel},
            TRAIN_PARAMS,
            A_accuracy=model_mse,
            T_accuracy=model_mse,
            eval_metric="mse",
            threshold=False,
            A_model_eq="noise",
            T_model_eq="noise",
        )

        dla_val = dla.getAmortizedLeakage(P, D, M)
        print(f"leakage for case {num+1} ({width=}): {dla}")
        print("______________________________________")
        print("______________________________________")
