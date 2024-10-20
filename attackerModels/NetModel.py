# NetModel.py
import torch
import torch.nn as nn
from attackerModels.ANN import simpleDenseModel

# Define the LSTM + ANN model
class LSTM_ANN_Model(nn.Module):
    def __init__(self, lstm_input_size, lstm_hidden_size, lstm_num_layers, lstm_bidirectional, ann_input_size, ann_output_size, num_ann_layers, ann_numFirst):
        super(LSTM_ANN_Model, self).__init__()

        # LSTM layer
        self.lstm = nn.LSTM(input_size=lstm_input_size,
                            hidden_size=lstm_hidden_size,
                            num_layers=lstm_num_layers,
                            bidirectional=lstm_bidirectional,
                            batch_first=True)

        self.ann = simpleDenseModel(input_dims=ann_input_size,
                                    output_dims=ann_output_size,
                                    num_layers=num_ann_layers,
                                    numFirst=ann_numFirst)

        self.output_layer = nn.Linear(ann_output_size, 3)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]

        ann_out = self.ann(lstm_out)

        output = self.output_layer(ann_out)
        output = torch.softmax(output, dim=-1)
        return output

if __name__ == "__main__":

    lstm_input_size = 100
    lstm_hidden_size = 128
    lstm_num_layers = 2
    lstm_bidirectional = True

    ann_input_size = lstm_hidden_size * 2 if lstm_bidirectional else lstm_hidden_size
    ann_output_size = 16
    num_ann_layers = 3
    ann_numFirst = 32

    model = LSTM_ANN_Model(lstm_input_size, lstm_hidden_size, lstm_num_layers, lstm_bidirectional, ann_input_size, ann_output_size, num_ann_layers, ann_numFirst)
    input_data = torch.randn(10, 50, lstm_input_size)
    output = model(input_data)

    print("Predicted probabilities:", output)
    predicted_class = torch.argmax(output, dim=-1)
    print("Predicted class:", predicted_class)