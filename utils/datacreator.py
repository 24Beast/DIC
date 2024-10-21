# Importing Libraries
import os
import pickle
import numpy as np
import pandas as pd
from typing import Union

# Type Hints
pathType = Union[str, os.PathLike]

# Data Class

class CaptionGenderDataset:
    
    def __init__(self, human_ann_file : pathType, model_ann_file: pathType) -> None:
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
    
    def processData(self):
        self.human_ann = {"img_id": [], "caption": []}
        self.model_ann = {"img_id": [], "caption": []}
        self.attribute_data = {"img_id" : [], "gender": [], "objects": []}
        
        for item in self.human_data:
            img_id = item["img_id"]
            gender = item["bb_gender"]
            objects = item["rmdup_object_list"]
            captions = item["caption_list"]
            
            self.attribute_data["img_id"].append(img_id)            
            self.attribute_data["gender"].append(gender)
            self.attribute_data["objects"].append(objects)
            
            self.human_ann["img_id"].extend([img_id] * len(captions))
            self.human_ann["img_id"].extend(captions)
        
        for item in self.model_data:
            img_id = item["img_id"]
            caption = item["pred"]
            
            self.model_ann["img_id"].append(img_id)
            self.model_ann["caption"].append(caption)
        
    def getData(self):
        return self.human_ann, self.model_ann, self.attribute_data