import os
import sys
import torch
import random
import argparse
import numpy as np
import pandas as pd
from DPIC import DPIC
from utils.text import CaptionProcessor
from utils.datacreator import CaptionGenderDataset
from attackerModels.NetModel import LSTM_ANN_Model

torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Print GPU details
print("GPU Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Current GPU:", torch.cuda.get_device_name(torch.cuda.current_device()))
    print("GPU Device Index:", torch.cuda.current_device())
    print("Total GPUs:", torch.cuda.device_count())

# Define thresholds for contextual DPIC
contextual_thresholds = [round(x * 0.05, 2) for x in range(11, 16)]

# Step 1: Define Gender Words and Token
masculine = [
    "man",
    "men",
    "male",
    "father",
    "gentleman",
    "gentlemen",
    "boy",
    "boys",
    "uncle",
    "husband",
    "actor",
    "prince",
    "waiter",
    "son",
    "he",
    "his",
    "him",
    "himself",
    "brother",
    "brothers",
    "guy",
    "guys",
    "emperor",
    "emperors",
    "dude",
    "dudes",
    "cowboy",
]
feminine = [
    "woman",
    "women",
    "female",
    "lady",
    "ladies",
    "mother",
    "girl",
    "girls",
    "aunt",
    "wife",
    "actress",
    "princess",
    "waitress",
    "daughter",
    "she",
    "her",
    "hers",
    "herself",
    "sister",
    "sisters",
    "queen",
    "queens",
    "pregnant",
]
gender_words = masculine + feminine
gender_token = "gender"


def processGender(data: pd.DataFrame) -> pd.DataFrame:
    m_cols = [item for item in data.columns if item in masculine]
    f_cols = [item for item in data.columns if item in feminine]
    data["M"] = data[m_cols].sum(axis=1)
    data["F"] = data[f_cols].sum(axis=1)
    data["M1"] = (data["M"] + 1e-5) / (data["M"] + data["F"] + 1e-5) > 0.5
    data["F1"] = (data["F"] + 1e-5) / (data["M"] + data["F"] + 1e-5) > 0.5
    return data[["caption", "M1", "F1"]]


def calculate_dpic(
    data_obj, processor, dpic_model, mode="non-contextual", threshold=0.5
):
    print(
        f"\nCalculating DPIC for mode: {mode}, Threshold: {threshold if threshold else 'N/A'}"
    )

    combined_data = data_obj.getDataCombined()
    human_ann = combined_data["caption_human"]
    model_ann = combined_data["caption_model"]
    print("\nLoaded Combined Dataset:")
    print(f"Total Samples: {len(combined_data)}")

    object_presence_df = data_obj.get_object_presence_df()
    obj_words = object_presence_df.columns.tolist()
    human_ann = data_obj.getLabelPresence(gender_words, human_ann)
    human_ann = processGender(human_ann)
    model_ann = data_obj.getLabelPresence(gender_words, model_ann)
    model_ann = processGender(model_ann)
    feat = combined_data.merge(object_presence_df, on="img_id").iloc[:, 4:].values
    feat = torch.tensor(feat).type(torch.float)

    print("\nPreprocessing Captions...")

    # Calculate DPIC Score
    dpic_score = dpic_model.getAmortizedLeakage(
        feat, human_ann, model_ann, num_trials=15, mask_mode="object"
    )
    print(f"\nDPIC Score for mode {mode}, Threshold {threshold}: {dpic_score}")
    return dpic_score


def main():
    parser = argparse.ArgumentParser(
        description="Test DPIC and Contextual DPIC calculations"
    )
    parser.add_argument(
        "--human_path", required=True, help="Path to human annotations pickle file"
    )
    parser.add_argument(
        "--model_path", required=True, help="Path to model annotations pickle file"
    )
    parser.add_argument(
        "--glove_path",
        required=True,
        help="Path to GloVe embeddings in word2vec format",
    )
    parser.add_argument(
        "--output_file",
        default="dpic_scores.csv",
        help="Output file to save DPIC scores",
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["contextual", "non-contextual"],
        help="Choose mode: 'contextual' or 'non-contextual'",
    )
    parser.add_argument(
        "--seed",
        default=0,
        help="Set random seed for the experiment. Helps ensure reproducability.",
    )
    args = parser.parse_args()

    # Setting random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Initialize objects
    data_obj = CaptionGenderDataset(args.human_path, args.model_path)
    object_presence_df = data_obj.get_object_presence_df()
    OBJ_WORDS = object_presence_df.columns.tolist()
    OBJ_TOKEN = "<obj>"
    NUM_OBJS = len(OBJ_WORDS)
    processor = CaptionProcessor(
        gender_words=gender_words,
        obj_words=[],
        glove_path=args.glove_path,
        tokenizer="nltk",
        gender_token=gender_token,
    )

    # Initialize DPIC Model
    dpic_model = DPIC(
        model_params={
            "attacker_class": LSTM_ANN_Model,
            "attacker_params": {
                "embedding_dim": 250,
                "pad_idx": 0,
                "lstm_hidden_size": 256,
                "lstm_num_layers": 2,
                "lstm_bidirectional": True,
                "ann_output_size": NUM_OBJS,
                "num_ann_layers": 5,
                "ann_numFirst": 64,
            },
        },
        train_params={
            "learning_rate": 0.01,
            "loss_function": "bce",
            "epochs": 50,
            "batch_size": 256,
        },
        gender_words=gender_words,
        obj_words=OBJ_WORDS,
        gender_token=gender_token,
        obj_token=OBJ_TOKEN,
        glove_path=args.glove_path,
        device=device,
        eval_metric="bce",
    )

    # Initialize results storage
    results = []

    # Calculate DPIC based on selected mode
    if args.mode == "non-contextual":
        non_contextual_dpic = calculate_dpic(
            data_obj, processor, dpic_model, mode="non-contextual"
        )
        results.append(
            {
                "mode": "non-contextual",
                "threshold": "N/A",
                "dpic_score_mean": non_contextual_dpic["Mean"].item(),
                "dpic_score_std_dev": non_contextual_dpic["std"].item(),
                "Number of Trials": non_contextual_dpic["num_trials"],
            }
        )

    elif args.mode == "contextual":
        for threshold in contextual_thresholds:
            contextual_dpic = calculate_dpic(
                data_obj, processor, dpic_model, mode="contextual", threshold=threshold
            )
            results.append(
                {
                    "mode": "contextual",
                    "threshold": threshold,
                    "dpic_score_mean": contextual_dpic["Mean"].item(),
                    "dpic_score_std_dev": contextual_dpic["std"].item(),
                    "Number of Trials": contextual_dpic["num_trials"],
                }
            )

    # Save results to CSV
    output_dir = "/".join(args.output_file.split("/")[:-1])
    if not (os.path.isdir(output_dir)):
        os.makedirs(output_dir)
    results_df = pd.DataFrame(results)
    results_df.to_csv(args.output_file, index=False)
    print(f"\nResults saved to {args.output_file}")


if __name__ == "__main__":
    main()
