# Importing Libraries
import os
import pickle
import nltk
import numpy as np
import pandas as pd
from typing import Union, Literal
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.preprocessing import MultiLabelBinarizer

nltk.download("wordnet")
pathType = Union[str, os.PathLike]
boolNum = Literal[0, 1]


# Helper Function
def checkWordPresence(word: str, sentence: str) -> int:
    return 1 if word in sentence.split(" ") else 0


# Data Class
class CaptionRaceDataset:
    def __init__(self, human_ann_file: pathType, model_ann_file: pathType) -> None:
        if not os.path.exists(human_ann_file) or not os.path.exists(model_ann_file):
            raise FileNotFoundError("Input files are missing.")

        self.human_ann_path = human_ann_file
        self.model_ann_path = model_ann_file
        print("Reading Annotation Files")
        self.human_data = self.read_pkl_file(self.human_ann_path)
        self.model_data = self.read_pkl_file(self.model_ann_path)
        self.wnl = WordNetLemmatizer()
        print("Processing Annotation Data")
        self.processData()

    @staticmethod
    def read_pkl_file(file_path: pathType) -> list[dict]:
        with open(file_path, "rb") as f:
            return pickle.load(f)

    def processData(self) -> None:
        self.mlb = MultiLabelBinarizer()
        self.human_ann = {"img_id": [], "caption": []}
        self.model_ann = {"img_id": [], "caption": []}
        self.attribute_data = {"img_id": [], "race": [], "objects": []}

        for item in self.human_data:
            self.attribute_data["img_id"].append(item["img_id"])
            self.attribute_data["race"].append(item["bb_skin"])
            self.attribute_data["objects"].append(item["rmdup_object_list"])
            self.human_ann["img_id"].extend(
                [item["img_id"]] * len(item["caption_list"])
            )
            self.human_ann["caption"].extend(item["caption_list"])

        for item in self.model_data:
            self.model_ann["img_id"].append(item["img_id"])
            self.model_ann["caption"].append(item["pred"])

        self.human_ann = pd.DataFrame(self.human_ann)
        self.model_ann = pd.DataFrame(self.model_ann)
        self.attribute_data = pd.DataFrame(self.attribute_data)

        objs = self.mlb.fit_transform(self.attribute_data["objects"])
        self.object_presence_df = pd.DataFrame(
            objs, columns=self.mlb.classes_, index=self.attribute_data["img_id"]
        )
        self.object_presence_df.fillna(0, inplace=True)
        race_mapping = {"Light": 0, "Dark": 1, "Both": 2, "": 2}
        self.attribute_data["race"] = self.attribute_data["race"].apply(
            lambda x: race_mapping.get(x, 2)  # Default to 2 for unknown values
        )

    def getData(self) -> list[pd.DataFrame]:
        human_merged = self.human_ann.merge(
            self.attribute_data.drop("objects", axis=1), on="img_id", how="left"
        )
        model_merged = self.model_ann.merge(
            self.attribute_data.drop("objects", axis=1), on="img_id", how="left"
        )
        return human_merged, model_merged

    def get_object_presence_df(self) -> pd.DataFrame:
        return self.object_presence_df

    def getDataCombined(self) -> pd.DataFrame:
        human_merged, model_merged = self.getData()
        return human_merged.merge(
            model_merged[["img_id", "caption"]],
            on="img_id",
            suffixes=["_human", "_model"],
        )

    def getLabelPresence(self, labels: list[str], captions: pd.Series) -> pd.DataFrame:
        new_labels = [self.wnl.lemmatize(label) for label in labels]
        new_captions = captions.apply(
            lambda x: " ".join([self.wnl.lemmatize(item) for item in x.split(" ")])
        )
        presence_df = pd.DataFrame({"caption": captions})
        for label in new_labels:
            presence_df[label] = new_captions.apply(
                lambda sentence: checkWordPresence(label, sentence)
            )
        return presence_df


if __name__ == "__main__":
    HUMAN_ANN_PATH = "./bias_data/Human_Ann/gender_obj_cap_mw_entries.pkl"
    MODEL_ANN_PATH = "./bias_data/Transformer/gender_val_transformer_cap_mw_entries.pkl"
    data_obj = CaptionRaceDataset(HUMAN_ANN_PATH, MODEL_ANN_PATH)
    human_ann, model_ann = data_obj.getData()
    object_presence_df = data_obj.get_object_presence_df()

    print("Human Annotations Sample:\n", human_ann.head())
    print("Model Annotations Sample:\n", model_ann.head())
    print("Object Presence DataFrame:\n", object_presence_df.head())
