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

# Helper Types
maskModeType = Literal["gender", "object"]


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
        glove_path=None,
        device="cpu",
    ) -> None:
        """
        Parameters
        ----------
        model_params : dict
            Dictionary of the following forms-
            {"attacker_class" : model_class,
             "attacker_params" : model_init_params}
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

    def calcLeak(
        self,
        feat: torch.tensor,
        data: torch.tensor,
        pred: torch.tensor,
        data_objs: np.array,
        pred_objs: np.array,
        apply_bayes: bool = True,
        normalized: bool = True,
        mask_mode: maskModeType = "gender",
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

        Returns
        -------
        leakage : torch.tensor
            Evaluated Leakage.

        """
        # Perform vocab equalization
        self.train(data, feat, "D")
        lambda_d = self.calcLambda(
            getattr(self, "attacker_D"), data, feat, data_objs, apply_bayes, mask_mode
        )
        self.train(pred, feat, "M")
        lambda_m = self.calcLambda(
            getattr(self, "attacker_M"), pred, feat, pred_objs, apply_bayes, mask_mode
        )
        print(f"{lambda_d=},\n{lambda_m=}")
        # TO DO: Process lambda values using bayes rule
        leakage_amp = lambda_m - lambda_d
        if normalized:
            leakage_amp = leakage_amp / (lambda_m + lambda_d)
        return leakage_amp

    def getProbsfromObjectOccurences(
        self, occurence_info: torch.tensor
    ) -> torch.tensor:
        val, inverse, counts = torch.unique(
            occurence_info, return_inverse=True, return_counts=True, dim=0
        )
        counts = counts / counts.sum()
        return counts[inverse]

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

    def getProbs(
        self,
        y: torch.tensor,
        y_pred: torch.tensor,
        mask_mode: maskModeType = "gender",
    ) -> torch.tensor:
        if mask_mode == "gender":
            args = y.argmax(axis=1)
            nums = np.arange(len(y))
            y_pred = y_pred.type(torch.float)
            probs = y_pred[nums, args]
        else:
            probs = (y_pred * y) + ((1 - y) * (1 - y_pred))
            probs = probs.prod(axis=1)
        return probs

    def calcLambda(
        self,
        model: torch.nn.Module,
        x: torch.tensor,
        y: torch.tensor,
        objs: np.array,
        apply_bayes: bool = True,
        mask_mode: maskModeType = "gender",
    ) -> torch.tensor:
        y_pred = model(x)
        y = y.type(torch.float)
        probs = self.getProbs(y, y_pred, mask_mode)
        if apply_bayes:
            probs_obj = self.getProbsfromObjectOccurences(objs)
            probs_attr = self.getProbsfromObjectOccurences(y)
            probs = (probs * probs_obj) / probs_attr
        return probs.mean()

    def defineModel(self) -> None:
        model_class = self.model_params["attacker_class"]
        model_params = self.model_params["attacker_params"]
        model_params["vocab_size"] = self.vocab_size
        self.attacker_D = model_class(**model_params)
        self.attacker_D.to(self.device)
        self.attacker_M = copy.deepcopy(self.attacker_D)
        self.attacker_M.to(self.device)

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
        self,
        model_captions: pd.Series,
        human_captions: pd.Series,
        mode: maskModeType = "gender",
    ) -> tuple[torch.tensor, torch.tensor]:  # type: ignore
        model_captions = self.capProcessor.maskWords(model_captions, mode=mode)
        human_captions = self.capProcessor.maskWords(human_captions, mode=mode)
        human_captions, model_captions = self.capProcessor.equalize_vocab(
            human_captions,
            model_captions,
            similarity_threshold=0.1,
        )
        model_vocab = self.capProcessor.build_vocab(model_captions)
        human_vocab = self.capProcessor.build_vocab(human_captions)
        print(f"{len(model_vocab)=}, {len(human_vocab)=}")
        self.vocab_size = max(len(model_vocab), len(human_vocab))
        model_cap = self.capProcessor.tokens_to_numbers(model_vocab, model_captions)
        human_cap = self.capProcessor.tokens_to_numbers(human_vocab, human_captions)
        return model_cap, human_cap

    def getAmortizedLeakage(
        self,
        feat: torch.tensor,  # Attribute
        data_frame: pd.DataFrame,  # Human Captions (straight from datacreator)
        pred_frame: pd.DataFrame,  # Model Captions (straight from datacreator)
        num_trials: int = 10,
        method: str = "mean",
        apply_bayes: bool = True,
        normalized: bool = True,
        mask_mode: maskModeType = "gender",
    ) -> tuple[torch.tensor, torch.tensor]:
        pred = pred_frame["caption"]
        data = data_frame["caption"]
        pred_objs = pred_frame.drop("caption", axis=1).to_numpy()
        data_objs = data_frame.drop("caption", axis=1).to_numpy()
        pred, data = self.captionPreprocess(pred, data, mask_mode)
        pred = pred.to(self.device)
        data = data.to(self.device)
        feat = feat.to(self.device)
        self.defineModel()
        vals = torch.zeros(num_trials)
        for i in range(num_trials):
            print(f"Working on Trial: {i}")
            vals[i] = self.calcLeak(
                feat,
                data,
                pred,
                data_objs,
                pred_objs,
                apply_bayes,
                normalized,
                mask_mode,
            )
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
    from attackerModels import LSTM_ANN_Model

    HUMAN_ANN_PATH = "./bias_data/Human_Ann/gender_obj_cap_mw_entries.pkl"
    MODEL_ANN_PATH = "./bias_data/Transformer/gender_val_transformer_cap_mw_entries.pkl"
    GLOVE_PATH = "./glove.6B.50d.w2vformat.txt"
    MASCULINE = [
        "man",
        "men",
        "male",
        "father",
        "gentleman",
        "boy",
        "uncle",
        "husband",
        "actor",
        "prince",
        "waiter",
        "he",
        "his",
        "him",
    ]
    FEMININE = [
        "woman",
        "women",
        "female",
        "mother",
        "lady",
        "girl",
        "aunt",
        "wife",
        "actress",
        "princess",
        "waitress",
        "she",
        "her",
        "hers",
    ]
    GENDER_WORDS = MASCULINE + FEMININE
    GENDER_TOKEN = "<unk>"
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_obj = CaptionGenderDataset(HUMAN_ANN_PATH, MODEL_ANN_PATH)
    ann_data = data_obj.getDataCombined()
    object_presence_df = data_obj.get_object_presence_df()
    OBJ_WORDS = object_presence_df.columns.tolist()
    OBJ_TOKEN = "<obj>"

    human_ann = ann_data["caption_human"]
    human_ann = data_obj.getLabelPresence(OBJ_WORDS, human_ann)
    model_ann = ann_data["caption_model"]
    gender = torch.tensor(ann_data["gender"]).reshape(-1, 1).type(torch.float)
    gender = torch.hstack([gender, 1 - gender])

    model_params = {
        "attacker_class": LSTM_ANN_Model,
        "attacker_params": {
            "embedding_dim": 250,
            "pad_idx": 0,
            "lstm_hidden_size": 128,
            "lstm_num_layers": 2,
            "lstm_bidirectional": True,
            "ann_output_size": 2,
            "num_ann_layers": 3,
            "ann_numFirst": 32,
        },
    }
    # Change format to intialize within LIC to allow vocab size to be passed later on.
    train_params = {
        "learning_rate": 0.01,
        "loss_function": "bce",
        "epochs": 100,
        "batch_size": 64,
    }

    DPIC_obj = DPIC(
        model_params,
        train_params,
        GENDER_WORDS,
        OBJ_WORDS,
        GENDER_TOKEN,
        OBJ_TOKEN,
        "bce",
        glove_path=GLOVE_PATH,
        device=DEVICE,
    )

    analysis_data = DPIC_obj.getAmortizedLeakage(
        gender, human_ann, model_ann, num_trials=5
    )
