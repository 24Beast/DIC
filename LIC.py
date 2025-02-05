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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Main class
class LIC:
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
        glove_path=None,
        device="cpu",
    ) -> None:
        self.model_params = model_params
        self.train_params = train_params
        self.model_attacker_trained = False
        self.threshold = threshold
        self.device = device

        self.loss_functions = {
            "mse": torch.nn.MSELoss(),
            "cross-entropy": torch.nn.CrossEntropyLoss(),
            "bce": torch.nn.BCELoss(),
        }
        self.eval_functions = {
            "accuracy": lambda y_pred, y: (y_pred == y).float().mean(),
            "mse": lambda y_pred, y: ((y_pred - y) ** 2).float().mean(),
            "bce": ModifiedBCELoss,
        }
        self.initEvalMetric(eval_metric)
        self.capProcessor = CaptionProcessor(
            gender_words,
            obj_words,
            gender_token=gender_token,
            obj_token=obj_token,
            glove_path=glove_path,
        )

    def initEvalMetric(self, metric: Union[Callable, str]) -> None:
        """
        Initialize evaluation metric for model evaluation.
        """
        if callable(metric):
            self.eval_metric = metric
        elif isinstance(metric, str):
            if metric in self.eval_functions:
                self.eval_metric = self.eval_functions[metric]
            else:
                raise ValueError(f"Metric {metric} not available.")
        else:
            raise ValueError("Invalid Metric Given.")

    def calcLeak(
        self,
        feat: torch.tensor,
        data: torch.tensor,
        pred: torch.tensor,
        normalized: bool = False,
    ) -> torch.tensor:
        self.train(data, feat, "D")
        lambda_d = self.calcLambda(getattr(self, "attacker_D"), data, feat)
        self.train(pred, feat, "M")
        lambda_m = self.calcLambda(getattr(self, "attacker_M"), pred, feat)
        print(f"{lambda_d=},\n{lambda_m=}")
        leakage_amp = lambda_m - lambda_d
        if normalized:
            leakage_amp = leakage_amp / (lambda_m + lambda_d)
        return leakage_amp

    def train(self, x, y, attacker_mode):
        self.defineModel()
        model = getattr(self, "attacker_" + attacker_mode)
        criterion = self.loss_functions[self.train_params["loss_function"]]
        optimizer = optim.Adam(
            model.parameters(), lr=self.train_params["learning_rate"]
        )
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        batches = math.ceil(len(x) / self.train_params["batch_size"])
        print(f"Training Activated for Mode: {attacker_mode}")

        for epoch in range(1, self.train_params["epochs"] + 1):
            perm = torch.randperm(x.shape[0], device=self.device)
            x, y = x[perm], y[perm]
            start, running_loss = 0, 0.0

            for _ in range(batches):
                x_batch = x[start : start + self.train_params["batch_size"]].to(self.device
                ).long()
                y_batch = y[start : start + self.train_params["batch_size"]].to(self.device)

                optimizer.zero_grad()
                outputs = model(x_batch)

                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                start += self.train_params["batch_size"]

            scheduler.step()

            if epoch % 5 == 0:
                print(f"Epoch {epoch}: Avg Loss = {running_loss / batches:.4f}")

    def calcLambda(self, model, x, y):
        model.eval()
        batch_size = self.train_params.get("batch_size", 32)

        y_pred_list = []
        total_samples = x.shape[0]

        with torch.no_grad():
            for start in range(0, total_samples, batch_size):
                end = min(start + batch_size, total_samples)
                x_batch = x[start:end].to(self.device)
                batch_pred = model(x_batch)
                y_pred_list.append(batch_pred.cpu())

        y_pred = torch.cat(y_pred_list, dim=0).to(self.device)

        y_pred = (y_pred > 0.5).float() if self.threshold else y_pred
        return self.eval_metric(y_pred, y.float())

    def defineModel(self):
        model_class = self.model_params["attacker_class"]
        model_params = self.model_params["attacker_params"]
        model_params["vocab_size"] = self.vocab_size
        self.attacker_D = model_class(**model_params).to(self.device)
        self.attacker_M = copy.deepcopy(self.attacker_D).to(self.device)

    def captionPreprocess(self, model_captions, human_captions):
        model_captions = self.capProcessor.maskWords(model_captions, mode="gender")
        human_captions = self.capProcessor.maskWords(human_captions, mode="gender")
        human_captions, model_captions = self.capProcessor.equalize_vocab(
            human_captions, model_captions, similarity_threshold=0.5
        )
        model_vocab = self.capProcessor.build_vocab(model_captions)
        human_vocab = self.capProcessor.build_vocab(human_captions)
        self.vocab_size = max(len(model_vocab), len(human_vocab))
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
    normalized: bool = False,
) -> tuple[torch.tensor, torch.tensor]:
        pred, data = self.captionPreprocess(pred, data)
        pred = pred.to(self.device)
        data = data.to(self.device)
        feat = feat.to(self.device)
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