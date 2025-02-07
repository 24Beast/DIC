import argparse
from utils.datacreator import CaptionGenderDataset
from utils.text import CaptionProcessor
from LIC import LIC
import torch
import pandas as pd
from tqdm import tqdm
import sys
import os
from attackerModels.NetModel import LSTM_ANN_Model, LSTM_RNN_Model

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

# Step 1: Define Gender Words and Token
masculine = [
    'man', 'men', 'male', 'father', 'gentleman', 'gentlemen', 'boy', 'boys', 'uncle', 'husband', 'actor', 'prince',
    'waiter', 'son', 'he', 'his', 'him', 'himself', 'brother', 'brothers', 'guy', 'guys', 'emperor', 'emperors', 
    'dude', 'dudes', 'cowboy'
]
feminine = [
    'woman', 'women', 'female', 'lady', 'ladies', 'mother', 'girl', 'girls', 'aunt', 'wife', 'actress', 'princess', 
    'waitress', 'daughter', 'she', 'her', 'hers', 'herself', 'sister', 'sisters', 'queen', 'queens', 'pregnant'
]
gender_words = masculine + feminine
gender_token = "gender"

def calculate_lic(data_obj, processor, lic_model, mode="non-contextual", threshold=0.5):
    print(f"\nCalculating LIC for mode: {mode}, Threshold: {threshold if threshold else 'N/A'}")

    combined_data = data_obj.getDataCombined()
    print("\nLoaded Combined Dataset:")
    print(f"Total Samples: {len(combined_data)}")

    # Extract Features
    human_ann = combined_data["caption_human"]
    model_ann = combined_data["caption_model"]
    feat = torch.tensor(combined_data["gender"].values, dtype=torch.float, device=device).reshape(-1, 1)

    print("\nPreprocessing Captions...")

    # Calculate LIC Score

    lic_score = lic_model.getAmortizedLeakage(feat, human_ann, model_ann, normalized=False)
    print(f"\nLIC Score for mode {mode}, Threshold {threshold}: {lic_score}")
    return lic_score

def main():
    parser = argparse.ArgumentParser(description="Test LIC and Contextual LIC calculations")
    parser.add_argument("--human_path", required=True, help="Path to human annotations pickle file")
    parser.add_argument("--model_path", required=True, help="Path to model annotations pickle file")
    parser.add_argument("--glove_path", required=True, help="Path to GloVe embeddings in word2vec format")
    parser.add_argument("--output_file", default="lic_scores.csv", help="Output file to save LIC scores")
    parser.add_argument("--mode", required=True, choices=["contextual", "non-contextual"], help="Choose mode: 'contextual' or 'non-contextual'")
    parser.add_argument("--model_type", required=True, choices=["lstm_ann", "lstm_rnn"], help="Choose model type: 'lstm_ann' or 'lstm_rnn'")
    args = parser.parse_args()

    # Initialize objects
    data_obj = CaptionGenderDataset(args.human_path, args.model_path)
    processor = CaptionProcessor(
        gender_words=gender_words, obj_words=[], glove_path=args.glove_path, tokenizer="nltk", gender_token=gender_token
    )

    model_class = LSTM_ANN_Model if args.model_type == "lstm_ann" else LSTM_RNN_Model

    # Initialize LIC
    lic_model = LIC(
        model_params={
            "attacker_class": model_class,
            "attacker_params": {
                "embedding_dim": 250,
                "pad_idx": 0,
                "lstm_hidden_size": 256,
                "lstm_num_layers": 2,
                "lstm_bidirectional": True,
                "ann_output_size": 1,
                "num_ann_layers": 5,
                "ann_numFirst": 64,
            },
        },
        train_params={
            "learning_rate": 0.01,
            "loss_function": "bce",
            "epochs": 100,
            "batch_size": 512,
        },
        gender_words=gender_words,
        obj_words=[],
        gender_token=gender_token,
        obj_token="obj",
        glove_path=args.glove_path,
        device=device,
        eval_metric = "bce",
        )

    # Initialize results storage
    results = []
    # Calculate LIC based on selected mode
    if args.mode == "non-contextual":
        non_contextual_lic = calculate_lic(data_obj, processor, lic_model, mode="non-contextual")
        results.append({"mode": "non-contextual", "threshold": "N/A", "lic_score": non_contextual_lic})

    elif args.mode == "contextual":
        for threshold in contextual_thresholds:
            contextual_lic = calculate_lic(data_obj, processor, lic_model, mode="contextual", threshold=threshold)
            results.append({"mode": "contextual", "threshold": threshold, "lic_score_mean": contextual_lic["Mean"].item(), "lic_score_std_dev": contextual_lic["std"].item(), "Number of Trials": contextual_lic["num_trials"]})

    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(args.output_file, index=False)
    print(f"\nResults saved to {args.output_file}")

if __name__ == "__main__":
    main()