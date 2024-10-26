# Importing Libraries
import copy
import math
import torch
import numpy as np
import pandas as pd
import torch.optim as optim
from typing import Callable, Union, Literal
from utils.losses import ModifiedBCELoss
from utils.text import CaptionProcessor


# Main class
class DPIC:
    def __init__(
        self,
        model_params: dict,
        train_params: dict,
        gender_words: list[str],
        obj_words: list[str],
        gender_token: str,
        obj_token: str,
        eval_metric: Union[Callable, str] = "mse",
        threshold=True,
    ) -> None:
        """
        Parameters
        ----------
        model_params : dict
            Dictionary of the following forms-
            {"attacker" : model}
        train_params : dict
            {
                "learning_rate": The learning rate hyperparameter,
                "loss_function": The loss function to be used.
                        Existing options: ["mse", "cross-entropy"],
                "epochs": Number of training epochs to be set,
                "batch_size: Number of batches per epoch
            }
        eval_metric : Union[Callable,str], optional
            Either a Callable of the form eval_metric(y_pred, y)
            or a string to utilize exiting methods.
            Existing options include ["accuracy"]
            The default is "mse".

        Returns
        -------
        None
            Initializes the class.

        """
        self.model_params = model_params
        self.train_params = train_params
        self.model_attacker_trained = False
        self.threshold = threshold

        self.loss_functions = {
            "mse": torch.nn.MSELoss(),
            "cross-entropy": torch.nn.CrossEntropyLoss(),
            "bce": torch.nn.BCELoss,
        }
        self.eval_functions = {
            "accuracy": lambda y_pred, y: (y_pred == y).float().mean(),
            "mse": lambda y_pred, y: ((y_pred - y) ** 2).float().mean(),
            "bce": ModifiedBCELoss,
        }
        self.initEvalMetric(eval_metric)
        self.defineModel()
        self.capProcessor = CaptionProcessor(
            gender_words, obj_words, gender_token=gender_token, obj_token=obj_token
        )

    def calcLeak(
        self,
        feat: torch.tensor,
        data: torch.tensor,
        pred: torch.tensor,
    ) -> torch.tensor:
        """
        Parameters
        ----------
        feat : torch.tensor
            Protected Attribute.
        data : torch.tensor
            Ground truth data.
        pred : torch.tensor
            Predicted Values.
        mode : Literal["AtoT","TtoA"]
            Sets Direction of calculation.

        Returns
        -------
        leakage : torch.tensor
            Evaluated Leakage.

        """
        # Perform vocab equalization
        pert_data = data
        self.train(feat, pert_data, "D")
        lambda_d = self.calcLambda(getattr(self, "attacker_D"), feat, pert_data)
        self.train(feat, pred, "M")
        lambda_m = self.calcLambda(getattr(self, "attacker_M"), feat, pred)
        print(f"{lambda_d=},\n{lambda_m=}")
        leakage_amp = lambda_m - lambda_d
        return leakage_amp

    def train(
        self,
        x: torch.tensor,
        y: torch.tensor,
        attacker_mode: str,
    ) -> torch.tensor:
        self.defineModel()
        model = getattr(self, "attacker_" + attacker_mode)
        criterion = self.loss_functions[self.train_params["loss_function"]]
        optimizer = optim.Adam(
            model.parameters(), lr=self.train_params["learning_rate"]
        )
        batches = math.ceil(len(x) / self.train_params["batch_size"])

        print(f"Training Activated for Mode: {attacker_mode}")

        # Training loop
        for epoch in range(1, self.train_params["epochs"] + 1):
            perm = torch.randperm(x.shape[0])
            x = x[perm]
            y = y[perm]
            start = 0
            running_loss = 0.0
            # print(batches)
            for batch_num in range(batches):
                x_batch = x[start : (start + self.train_params["batch_size"])]
                y_batch = y[start : (start + self.train_params["batch_size"])]

                optimizer.zero_grad()
                # Forward pass
                outputs = model(x_batch)
                # print(f"{outputs=}\n{y_batch=}")
                loss = criterion(outputs, y_batch)
                # print(f"{loss.item()=}")

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                start += self.train_params["batch_size"]
                running_loss += loss.item()

            avg_loss = running_loss / batches
            if epoch % 10 == 0:
                print(f"\rCurrent Epoch {epoch}: Loss = {avg_loss}", end="")

        print("\nModel training completed")

    def calcLambda(
        self, model: torch.nn.Module, x: torch.tensor, y: torch.tensor
    ) -> torch.tensor:
        y_pred = model(x)
        if self.threshold:
            y_pred = y_pred > 0.5
        y = y.type(torch.float)
        y_pred = y_pred.type(torch.float)
        return self.eval_metric(y_pred, y)

    def defineModel(self) -> None:
        self.attacker_D = self.model_params["attacker"]
        self.attacker_M = copy.deepcopy(self.attacker_D)

    def initEvalMetric(self, metric: Union[Callable, str]) -> None:
        if callable(metric):
            self.eval_metric = metric
        elif type(metric) == str:
            if metric in self.eval_functions.keys():
                self.eval_metric = self.eval_functions[metric]
            else:
                raise ValueError("Metric Option given is unavailable.")
        else:
            raise ValueError("Invalid Metric Given.")

    def captionPreprocess(
        self, model_captions: pd.Series, human_captions: pd.Series
    ) -> tuple(torch.tensor, torch.tensor):
        model_captions, human_captions = self.capProcessor.equalize_vocab(
            model_captions, human_captions
        )
        model_vocab = self.capProcessor.build_vocab(model_captions)
        human_vocab = self.capProcessor.build_vocab(human_captions)
        model_cap = self.capProcessor.tokens_to_numbers(model_vocab, model_captions)
        human_cap = self.capProcessor.tokens_to_numbers(human_vocab, human_captions)
        return model_cap, human_cap

    def getAmortizedLeakage(
        self,
        feat: torch.tensor,
        data: pd.Series,
        pred: pd.Series,
        num_trials: int = 10,
        method: str = "mean",
        normalized: bool = True,
    ) -> tuple[torch.tensor, torch.tensor]:
        pred, data = self.captionPreprocess(pred, data)
        vals = torch.zeros(num_trials)
        for i in range(num_trials):
            print(f"Working on Trial: {i}")
            vals[i] = self.calcLeak(feat, data, pred, normalized)
            print(f"Trial {i} val: {vals[i]}")
        if method == "mean":
            return {
                "Mean": torch.mean(vals),
                "std": torch.std(vals),
                "num_trials": num_trials,
            }
        elif method == "median":
            return {
                "Median": torch.median(vals),
                "std": torch.std(vals),
                "num_trials": num_trials,
            }
        else:
            raise ValueError("Invalid Method given for Amortization.")


if __name__ == "__main__":
    from utils.datacreator import CaptionGenderDataset
