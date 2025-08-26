import os
import sys
import torch
import random
import argparse
import numpy as np
import pandas as pd
from LIC import LIC
from utils.text import CaptionProcessor
from utils.datacreator_race import CaptionRaceDataset
from attackerModels.NetModel import LSTM_ANN_Model, RNN_ANN_Model

torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sys.path.append("/home/nshah96/DIC")
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

print("GPU Available:", torch.cuda.is_available())
# Print the currently active GPU
if torch.cuda.is_available():
    print("Current GPU:", torch.cuda.get_device_name(torch.cuda.current_device()))
    print("GPU Device Index:", torch.cuda.current_device())
    print("Total GPUs:", torch.cuda.device_count())

# Define thresholds for contextual LIC
contextual_thresholds = [round(x * 0.05, 2) for x in range(11, 16)]

# Step 1: Define Race Words and Token
race_words = [
    "white",
    "caucasian",
    "black",
    "african",
    "asian",
    "latino",
    "latina",
    "latinx",
    "hispanic",
    "native",
    "indigenous",
]
race_token = "race"


def calculate_lic(data_obj, processor, lic_model, mode="non-contextual", threshold=0.5):
    print(
        f"\nCalculating LIC for mode: {mode}, Threshold: {threshold if threshold else 'N/A'}"
    )

    combined_data = data_obj.getDataCombined()
    print("\nLoaded Combined Dataset:")
    print(f"Total Samples: {len(combined_data)}")

    # Extract Features
    human_ann = combined_data["caption_human"]
    model_ann = combined_data["caption_model"]
    feat = torch.tensor(
        combined_data["race"].values, dtype=torch.float, device=device
    ).reshape(-1, 1)

    print("\nPreprocessing Captions...")

    # Calculate LIC Score

    lic_score = lic_model.getAmortizedLeakage(
        feat, human_ann, model_ann, normalized=False
    )
    print(f"\nLIC Score for mode {mode}, Threshold {threshold}: {lic_score}")
    return lic_score


def main():
    parser = argparse.ArgumentParser(
        description="Test LIC and Contextual LIC calculations for race"
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
        "--output_file", default="lic_scores_race.csv", help="Output file to save LIC scores"
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["contextual", "non-contextual"],
        help="Choose mode: 'contextual' or 'non-contextual'",
    )
    parser.add_argument(
        "--use_rnn", action="store_true", help="Use RNN instead of LSTM"
    )
    parser.add_argument(
        "--bidirectional", action="store_true", help="Use bidirectional LSTM/RNN"
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
    data_obj = CaptionRaceDataset(args.human_path, args.model_path)
    processor = CaptionProcessor(
        gender_words=race_words,
        obj_words=[],
        glove_path=args.glove_path,
        tokenizer="nltk",
        gender_token=race_token,
    )

    if args.use_rnn:
        model_type = RNN_ANN_Model
        model_params = {
            "embedding_dim": 250,
            "pad_idx": 0,
            "rnn_hidden_size": 256,
            "rnn_num_layers": 2,
            "rnn_bidirectional": args.bidirectional,
            "ann_output_size": 1,
            "num_ann_layers": 5,
            "ann_numFirst": 64,
        }

    else:
        model_type = LSTM_ANN_Model
        model_params = {
            "embedding_dim": 250,
            "pad_idx": 0,
            "lstm_hidden_size": 256,
            "lstm_num_layers": 2,
            "lstm_bidirectional": args.bidirectional,
            "ann_output_size": 1,
            "num_ann_layers": 5,
            "ann_numFirst": 64,
        }

    # Initialize LIC
    lic_model = LIC(
        model_params={
            "attacker_class": model_type,
            "attacker_params": model_params,
        },
        train_params={
            "learning_rate": 0.01,
            "loss_function": "bce",
            "epochs": 50,
            "batch_size": 1024,
        },
        gender_words=race_words,
        obj_words=[],
        gender_token=race_token,
        obj_token="obj",
        glove_path=args.glove_path,
        device=device,
        eval_metric="bce",
    )

    # Initialize results storage
    results = []

    if args.mode == "non-contextual":
        non_contextual_lic = calculate_lic(
            data_obj, processor, lic_model, mode="non-contextual"
        )
        results.append(
            {
                "mode": "non-contextual",
                "threshold": "N/A",
                "lic_score_mean": non_contextual_lic["Mean"].item(),
                "lic_score_std_dev": non_contextual_lic["std"].item(),
                "Number of Trials": non_contextual_lic["num_trials"],
            }
        )

    elif args.mode == "contextual":
        for threshold in contextual_thresholds:
            contextual_lic = calculate_lic(
                data_obj, processor, lic_model, mode="contextual", threshold=threshold
            )
            results.append(
                {
                    "mode": "contextual",
                    "threshold": threshold,
                    "lic_score_mean": contextual_lic["Mean"].item(),
                    "lic_score_std_dev": contextual_lic["std"].item(),
                    "Number of Trials": contextual_lic["num_trials"],
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