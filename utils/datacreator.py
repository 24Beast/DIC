# Importing Libraries
import os
import pickle
import numpy as np
import pandas as pd
from typing import Union
from sklearn.preprocessing import MultiLabelBinarizer

# Type Hints
pathType = Union[str, os.PathLike]


# Data Class
class CaptionGenderDataset:

    def __init__(self, human_ann_file: pathType, model_ann_file: pathType) -> None:
        self.human_ann_path = human_ann_file
        self.model_ann_path = model_ann_file
        print("Reading Annontation Files")
        self.human_data = self.read_pkl_file(self.human_ann_path)
        self.model_data = self.read_pkl_file(self.model_ann_path)
        print("Processing Annontation Data")
        self.processData()

    @staticmethod
    def read_pkl_file(file_path: pathType) -> list[dict]:
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        return data

    def processData(self) -> None:
        self.mlb = MultiLabelBinarizer()
        self.human_ann = {"img_id": [], "caption": []}
        self.model_ann = {"img_id": [], "caption": []}
        self.attribute_data = {"img_id": [], "gender": [], "objects": []}

        for item in self.human_data:
            img_id = item["img_id"]
            gender = item["bb_gender"]
            objects = item["rmdup_object_list"]
            captions = item["caption_list"]

            self.attribute_data["img_id"].append(img_id)
            self.attribute_data["gender"].append(gender)
            self.attribute_data["objects"].append(objects)

            self.human_ann["img_id"].extend([img_id] * len(captions))
            self.human_ann["caption"].extend(captions)

        for item in self.model_data:
            img_id = item["img_id"]
            caption = item["pred"]

            self.model_ann["img_id"].append(img_id)
            self.model_ann["caption"].append(caption)

        self.human_ann = pd.DataFrame(self.human_ann)
        self.model_ann = pd.DataFrame(self.model_ann)
        self.attribute_data = pd.DataFrame(self.attribute_data)
        objs = self.mlb.fit_transform(self.attribute_data["objects"])
        self.attribute_data[self.mlb.classes_] = objs

    def getData(self) -> list[pd.DataFrame]:
        return self.human_ann.merge(self.attribute_data), self.model_ann.merge(
            self.attribute_data
        )


"""
Note: Use MultiLabelBinarizer to convert objects in attribute_data
"""


if __name__ == "__main__":
    HUMAN_ANN_PATH = "../bias_data/Human_Ann/gender_obj_cap_mw_entries.pkl"
    MODEL_ANN_PATH = (
        "../bias_data/Transformer/gender_val_transformer_cap_mw_entries.pkl"
    )
    data_obj = CaptionGenderDataset(HUMAN_ANN_PATH, MODEL_ANN_PATH)
    human_ann, model_ann = data_obj.getData()
