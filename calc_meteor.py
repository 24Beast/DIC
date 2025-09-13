import nltk
import numpy as np
import pandas as pd
import torch
from utils.text import CaptionProcessor
from utils.datacreator import CaptionGenderDataset
from nltk.translate.meteor_score import meteor_score

# nltk.download('wordnet')


GLOVE_PATH = "./glove.6B.50d.w2vformat.txt"
SUB_MODEL = (
    "bert"  # Use glove for fasttext as well! IDK why it is not working as expected
)
DEVICE = torch.device("cpu")  # Force CPU to avoid CUDA-related segfaults

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

HUMAN_ANN_PATH = "./data/gender_obj_cap_mw_entries.pkl"
MODEL_ANN_PATH = "./data/new_models/no_masking/bakllava.pkl"


data_obj = CaptionGenderDataset(HUMAN_ANN_PATH, MODEL_ANN_PATH)
ann_data = data_obj.getDataCombined()
object_presence_df = data_obj.get_object_presence_df()
OBJ_WORDS = object_presence_df.columns.tolist()
OBJ_TOKEN = "<obj>"
NUM_OBJS = len(OBJ_WORDS)


capProcessor = CaptionProcessor(
    GENDER_WORDS,
    OBJ_WORDS,
    gender_token=GENDER_TOKEN,
    obj_token=OBJ_TOKEN,
    glove_path=GLOVE_PATH,
    model_type=SUB_MODEL,
    device=DEVICE,
)


def captionPreprocess(
    human_captions: pd.Series,
    model_captions: pd.Series,
    mode="gender",
    similarity_threshold=0.5,
    maskType="contextual",
):  # type: ignore
    model_captions = capProcessor.maskWords(model_captions, mode=mode)
    human_captions = capProcessor.maskWords(human_captions, mode=mode)
    human_captions, model_captions = capProcessor.equalize_vocab(
        human_captions,
        model_captions,
        similarity_threshold=similarity_threshold,
        maskType=maskType,
        bidirectional=False,
    )
    return human_captions, model_captions


def calculate_meteor(refs, candidate):
    return meteor_score([item.split() for item in refs], candidate.split())


human_ann = ann_data["caption_human"]
model_ann = ann_data["caption_model"].iloc[::5]

human_cap, model_cap = captionPreprocess(human_ann, model_ann)

vals = np.zeros(len(model_ann))

for i in range(len(model_ann)):
    vals[i] = calculate_meteor([human_ann[i]], human_cap[i])

print(f"{vals.mean()=}")
