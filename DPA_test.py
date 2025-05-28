# Importing Libraries
import math
import copy
import torch
import pandas as pd
from typing import Callable, Union, Literal
from attackerModels.NetModel import simpleDenseModel
from utils.datacreator import CaptionGenderDataset


# Defining Constants
HUMAN_ANN_PATH = "./bias_data/Human_Ann/gender_obj_cap_mw_entries.pkl"
MODEL_ANN_PATH = "./bias_data/Att2In_FC/gender_val_fc_cap_mw_entries.pkl"
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
OBJ_TOKEN = "<obj>"
NUM_GENDERS = 2
NUM_EPOCHS = 50
NUM_TRIALS = 4


# Helper Functions
def processGender(data: pd.DataFrame) -> pd.DataFrame:
    m_cols = [item for item in data.columns if item in MASCULINE]
    f_cols = [item for item in data.columns if item in FEMININE]
    data["M"] = data[m_cols].sum(axis=1)
    data["F"] = data[f_cols].sum(axis=1)
    data["M1"] = (data["M"] + 1e-5) / (data["M"] + data["F"] + 1e-5) > 0.5
    data["F1"] = (data["F"] + 1e-5) / (data["M"] + data["F"] + 1e-5) > 0.5
    return data[["caption", "M1", "F1"]]


bce_loss = torch.nn.BCELoss()


def ModifiedBCELoss(y_pred, y):
    return 1 / bce_loss(y_pred, y)


# Main class
class DLA:
    def __init__(
        self,
        model_params: dict,
        train_params: dict,
        model_acc: float,
        eval_metric: Union[Callable, str] = "mse",
        threshold=True,
    ) -> None:
        """
        Parameters
        ----------
        model_params : dict
            Dictionary of the following forms-
            {"attacker_AtoT" : model_AT, "attacker_TtoA" : model_TA}
        train_params : dict
            {
                "AtoT":
                    {
                        "learning_rate": The learning rate hyperparameter,
                        "loss_function": The loss function to be used.
                                Existing options: ["mse", "cross-entropy"],
                        "epochs": Number of training epochs to be set,
                        "batch_size: Number of batches per epoch
                    },
                "TtoA": {same format as AtoT}
            }
        model_acc : float
            The accuracy of the model being tested for quality equalization.
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
        self.model_acc = model_acc

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
        self.defineModel()

    def calcLeak(
        self,
        feat: torch.tensor,
        data: torch.tensor,
        pred: torch.tensor,
        mode: Literal["AtoT", "TtoA"],
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
        pert_data = self.permuteData(data)
        self.train(feat, pert_data, "D_" + mode)
        lambda_d = self.calcLambda(getattr(self, "attacker_D_" + mode), feat, pert_data)
        self.train(feat, pred, "M_" + mode)
        lambda_m = self.calcLambda(getattr(self, "attacker_M_" + mode), feat, pred)
        print(f"{lambda_d=},\n{lambda_m=}")
        leakage = (lambda_m - lambda_d) / (lambda_m + lambda_d)
        return leakage

    def train(
        self,
        x: torch.tensor,
        y: torch.tensor,
        attacker_mode: str,
    ) -> torch.tensor:
        self.defineModel()
        model = getattr(self, "attacker_" + attacker_mode)
        criterion = self.loss_functions[self.train_params["loss_function"]]
        optimizer = torch.optim.Adam(
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
        return self.eval_metric(y_pred, y)

    def defineModel(self) -> None:
        if type(self.model_params.get("attacker_AtoT", None)) == None:
            raise Exception("attacker_AtoT Model Missing!")
        if type(self.model_params.get("attacker_TtoA", None)) == None:
            raise Exception("attacker_TtoA Model Missing!")
        self.attacker_D_AtoT = self.model_params["attacker_AtoT"]
        self.attacker_M_AtoT = copy.deepcopy(self.attacker_D_AtoT)
        self.attacker_D_TtoA = self.model_params["attacker_TtoA"]
        self.attacker_M_TtoA = copy.deepcopy(self.attacker_D_TtoA)

    def permuteData(self, data: torch.tensor) -> torch.tensor:
        """
        Currently assumes ground truth data to be binary values in a pytorch tensor.
        Should work for any NxM type array.

        Parameters
        ----------
        data : torch.tensor
            Original ground truth data.

        Returns
        -------
        new_data : torch.tensor
            Randomly pertubed data for quality equalization.
        """
        if self.model_acc > 1:
            self.model_acc = self.model_acc / 100
        num_observations = data.shape[0]
        rand_vect = torch.zeros((num_observations, 1))
        rand_vect[: int(self.model_acc * num_observations)] = 1
        rand_vect = rand_vect[torch.randperm(num_observations)]
        new_data = rand_vect * (data) + (1 - rand_vect) * (1 - data)
        return new_data

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

    def getAmortizedLeakage(
        self,
        feat: torch.tensor,
        data: torch.tensor,
        pred: torch.tensor,
        mode: Literal["AtoT", "TtoA"],
        num_trials: int = 10,
        method: str = "mean",
    ) -> tuple[torch.tensor, torch.tensor]:
        vals = torch.zeros(num_trials)
        for i in range(num_trials):
            print(f"Working on Trial: {i}")
            vals[i] = self.calcLeak(feat, data, pred, mode)
            print(f"Trial {i} val: {vals[i]}")
        if method == "mean":
            return torch.mean(vals), torch.std(vals)
        elif method == "median":
            return torch.median(vals), torch.std(vals)
        else:
            raise ValueError("Invalid Method given for Amortization.")

    def calcBidirectional(
        self,
        A: torch.tensor,
        T: torch.tensor,
        A_pred: torch.tensor,
        T_pred: torch.tensor,
        num_trials: int = 10,
        method: str = "mean",
    ) -> tuple[tuple[torch.tensor, torch.tensor], tuple[torch.tensor, torch.tensor]]:
        AtoT_vals = self.getAmortizedLeakage(A, T, T_pred, "AtoT", num_trials, method)
        TtoA_vals = self.getAmortizedLeakage(T, A, A_pred, "TtoA", num_trials, method)
        return (AtoT_vals, TtoA_vals)


data_obj = CaptionGenderDataset(HUMAN_ANN_PATH, MODEL_ANN_PATH)
ann_data = data_obj.getDataCombined()
object_presence_df = data_obj.get_object_presence_df()
OBJ_WORDS = object_presence_df.columns.tolist()
NUM_OBJS = len(OBJ_WORDS)

combined_data = data_obj.getDataCombined()
human_ann = combined_data["caption_human"]
model_ann = combined_data["caption_model"]
print("\nLoaded Combined Dataset:")
print(f"Total Samples: {len(combined_data)}")


human_ann = data_obj.getLabelPresence(GENDER_WORDS, human_ann)
gh_ann = processGender(human_ann)
A_D = gh_ann[["M1", "F1"]].values * 1
A_D = A_D / A_D.sum(axis=1).reshape(-1, 1)
A_D = torch.tensor(A_D, dtype=torch.float)
model_ann = data_obj.getLabelPresence(GENDER_WORDS, model_ann)
gm_ann = processGender(model_ann)
A_M = gm_ann[["M1", "F1"]].values * 1
A_M = A_M / A_M.sum(axis=1).reshape(-1, 1)
A_M = torch.tensor(A_M, dtype=torch.float)
feat = combined_data.merge(object_presence_df, on="img_id").iloc[:, 4:].values
T = torch.tensor(feat).type(torch.float)

human_ann = data_obj.getLabelPresence(OBJ_WORDS, human_ann["caption"])
T_D = torch.tensor(human_ann.values[:, 1:].astype(float), dtype=torch.float)
model_ann = data_obj.getLabelPresence(OBJ_WORDS, model_ann["caption"])
T_M = torch.tensor(model_ann.values[:, 1:].astype(float), dtype=torch.float)


A = torch.tensor(combined_data["gender"].values, dtype=torch.float).reshape(-1, 1)
A = torch.hstack([A, 1 - A])


attackerModel_AtoT = simpleDenseModel(
    NUM_GENDERS, NUM_OBJS, 2, numFirst=4, activations=["sigmoid", "sigmoid", "sigmoid"]
)

attackerModel_TtoA = simpleDenseModel(
    NUM_OBJS, NUM_GENDERS, 2, numFirst=4, activations=["sigmoid", "sigmoid", "sigmoid"]
)

# Parameter Initialization
leakage = DLA(
    {"attacker_AtoT": attackerModel_AtoT, "attacker_TtoA": attackerModel_TtoA},
    {
        "learning_rate": 0.05,
        "loss_function": "bce",
        "epochs": NUM_EPOCHS,
        "batch_size": 2048,
    },
    1.0,
    "bce",
    threshold=False,
)

leak_AtoT = leakage.getAmortizedLeakage(A, T_D, T_M, "AtoT", num_trials=NUM_TRIALS)

leak_TtoA = leakage.getAmortizedLeakage(T, A_D, A_M, "TtoA", num_trials=NUM_TRIALS)

print(f"{leak_AtoT=}")
print(f"{leak_TtoA=}")
