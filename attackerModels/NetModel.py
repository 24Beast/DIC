import torch
import torch.nn as nn
import argparse
from attackerModels.ANN import simpleDenseModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"


# Define the LSTM + ANN model
class LSTM_ANN_Model(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        pad_idx,
        lstm_hidden_size,
        lstm_num_layers,
        lstm_bidirectional,
        ann_output_size,
        num_ann_layers,
        ann_numFirst,
    ):
        super(LSTM_ANN_Model, self).__init__()

        # Embedding layer
        self.embed = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        # LSTM layer with dropout
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            bidirectional=lstm_bidirectional,
            batch_first=True,
            dropout=0.3,
        )

        self.ann = simpleDenseModel(
            input_dims=lstm_hidden_size * 2 if lstm_bidirectional else lstm_hidden_size,
            output_dims=ann_output_size,
            num_layers=num_ann_layers,
            numFirst=ann_numFirst,
        )

        self.lastAct = nn.Sigmoid()
        if ann_output_size > 1:
            self.lastAct = nn.Softmax()

    def forward(self, x):

        x = x.to(device)

        # Embedding
        x = self.embed(x)
        assert (
            len(x.shape) == 3
        ), f"Expected input shape [batch_size, seq_len], but got {x.shape}"

        # LSTM
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]  # Take the last hidden state

        # ANN
        ann_out = self.ann(lstm_out)

        # Output layer and log-softmax
        ann_out = self.lastAct(ann_out)
        return ann_out

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class RNN_ANN_Model(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        pad_idx,
        rnn_hidden_size,
        rnn_num_layers,
        rnn_bidirectional,
        ann_output_size,
        num_ann_layers,
        ann_numFirst,
    ):
        super(RNN_ANN_Model, self).__init__()

        # Embedding layer
        self.embed = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        # Simple RNN layer with dropout
        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            bidirectional=rnn_bidirectional,
            batch_first=True,
            dropout=0.3,
        )

        # ANN layer after RNN
        self.ann = simpleDenseModel(
            input_dims=rnn_hidden_size * 2 if rnn_bidirectional else rnn_hidden_size,
            output_dims=ann_output_size,
            num_layers=num_ann_layers,
            numFirst=ann_numFirst,
        )

        # Activation function
        self.lastAct = nn.Sigmoid()
        if ann_output_size > 1:
            self.lastAct = nn.Softmax()

    def forward(self, x):
        x = x.to(device)

        # Embedding
        x = self.embed(x)
        assert (
            len(x.shape) == 3
        ), f"Expected input shape [batch_size, seq_len], but got {x.shape}"

        # RNN
        rnn_out, _ = self.rnn(x)
        rnn_out = rnn_out[:, -1, :]  # Take the last hidden state

        # ANN
        ann_out = self.ann(rnn_out)

        # Final activation
        ann_out = self.lastAct(ann_out)
        return ann_out

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def get_parser():
    """
    CLI parser for training parameters and file paths.
    """
    parser = argparse.ArgumentParser(description="Train and Save LSTM + ANN Model")
    parser.add_argument(
        "--vocab_size", type=int, required=True, help="Size of the vocabulary"
    )
    parser.add_argument(
        "--embedding_dim", type=int, default=100, help="Embedding dimension"
    )
    parser.add_argument(
        "--pad_idx", type=int, default=0, help="Padding index for embeddings"
    )
    parser.add_argument(
        "--lstm_hidden_size", type=int, default=128, help="Hidden size of the LSTM"
    )
    parser.add_argument(
        "--lstm_num_layers", type=int, default=2, help="Number of LSTM layers"
    )
    parser.add_argument(
        "--lstm_bidirectional", action="store_true", help="Use bidirectional LSTM"
    )
    parser.add_argument(
        "--output_size", type=int, required=True, help="Output size of ANN"
    )
    parser.add_argument(
        "--num_ann_layers", type=int, default=3, help="Number of layers in ANN"
    )
    parser.add_argument(
        "--ann_numFirst",
        type=int,
        default=32,
        help="Number of units in the first ANN layer",
    )
    parser.add_argument(
        "--save_model_path", required=True, help="Path to save the trained model"
    )
    return parser


def main(args):
    # Define LSTM + ANN model
    if args.model_type == "LSTM_ANN":
        model = LSTM_ANN_Model(
            vocab_size=args.vocab_size,
            embedding_dim=args.embedding_dim,
            pad_idx=args.pad_idx,
            lstm_hidden_size=args.lstm_hidden_size,
            lstm_num_layers=args.lstm_num_layers,
            lstm_bidirectional=args.lstm_bidirectional,
            ann_output_size=args.ann_output_size,
            num_ann_layers=args.num_ann_layers,
            ann_numFirst=args.ann_numFirst,
        ).to(device)

    else:
        model = RNN_ANN_Model(
            vocab_size=args.vocab_size,
            embedding_dim=args.embedding_dim,
            pad_idx=args.pad_idx,
            lstm_hidden_size=args.lstm_hidden_size,
            lstm_num_layers=args.lstm_num_layers,
            lstm_bidirectional=args.lstm_bidirectional,
            output_size=args.output_size,
        ).to(device)

    # Save model
    torch.save(model.state_dict(), args.save_model_path)
    print(f"Model saved to {args.save_model_path}")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
