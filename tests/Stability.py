# Importing Libraries
import copy
import math
import torch
import numpy as np
import torch.optim as optim
from typing import Callable, Union, Literal
from DLA import DLA


def RMSE(x, y):
    if len(x) != len(y):
        raise ValueError(
            f"Length of x and y must be same! Current shapes: {x.shape=} & {y.shape=}"
        )
    return (torch.sum((x - y) ** 2) ** 0.5) / len(x)


if __name__ == "__main__":
    # Test case
    import os
    import sys
    import json

    #    from ..DLA import DLA
    sys.path.append("../")
    from Leakage import Leakage
    from attackerModels.ANN import simpleDenseModel

    # Data Initialization
    from utils.datacreator import StabilityExp

    NUM_SAMPLES = 16384
    DATA_ERROR_W = 0.4
    MODEL_ERROR_W = 0.02
    POLY_POW = 2
    DATA_RANGE = (1, 5)
    ATTACKER_WIDTHS = [i for i in range(25, 501, 25)]
    ATTACKER_DEPTHS = [i for i in range(0, 10)]
    OUTFILE = "results/Stability.json"

    P, D, M = StabilityExp(
        NUM_SAMPLES, DATA_ERROR_W, MODEL_ERROR_W, POLY_POW, DATA_RANGE
    )
    P = torch.tensor(P, dtype=torch.float).reshape(-1, 1)
    D = torch.tensor(D, dtype=torch.float).reshape(-1, 1)
    M = torch.tensor(M, dtype=torch.float).reshape(-1, 1)

    # Calculating Params
    model_mse = RMSE(M, D)
    leakages = {}

    num = 0
    for depth in ATTACKER_DEPTHS:
        for width in ATTACKER_WIDTHS:
            print(f"Working on Iteration {num}", flush=True)
            # Attacker Model Initialization
            attackerModel = simpleDenseModel(
                1, width, 1, numFirst=1, activations=["relu", "relu", "relu"]
            )

            num_params = attackerModel.count_params()

            # Parameter Initialization
            leakage = Leakage(
                {"attacker_D": attackerModel, "sameModel": True},
                {
                    "learning_rate": 0.005,
                    "loss_function": "mse",
                    "epochs": 200,
                    "batch_size": 256,
                },
                model_mse,
                "noise",
                "mse",
                threshold=False,
            )

            leak = leakage.getAmortizedLeakage(P, D, M)
            num += 1
            print(
                f"leakage for case {num} ({width=}) ({depth=}) ({num_params=}): {leak}"
            )
            print("______________________________________")
            print("______________________________________")
            leakages[num] = {
                "width": width,
                "depth": depth,
                "num_params": num_params,
                "data": leak,
            }

    print("Saving results!")
    save_dir = "/".join(OUTFILE.split("/")[:-1])
    os.makedirs(save_dir, exist_ok=True)
    with open(OUTFILE, "w") as f:
        json.dump(leakages, f)
