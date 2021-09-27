import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from modules.utils.pad_sequences import get_seq_length_from_padded_seq


class ComparisonLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=256, output_size=1, num_layers=1, num_targets=1):
        """
        Init of LSTM class for comparison of NN vs LSTM to determine the need of time series
        @param input_size: number of input units
        @param hidden_size: number of hidden units [defaults to 256]
        @param output_size: number of output units [defaults to 1]
        @param num_layers: number of layers [defaults to 1]
        """
        super(ComparisonLSTM, self).__init__()

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

        self.output_layers = nn.ModuleList([nn.Linear(hidden_size, output_size) for i in range(num_targets)])

        self.h_c = None

    def forward(self, features, h_c=None, apply_activation=False):
        n_timesteps = features.shape[1]
        outputs = []

        seq_lengths = get_seq_length_from_padded_seq(features.clone().detach().cpu().numpy())
        features = pack_padded_sequence(features, seq_lengths, batch_first=True, enforce_sorted=False)
        if h_c is None:
            intermediate, h_c = self.lstm(features)
        else:
            h, c = h_c
            intermediate, h_c = self.lstm(features, h, c)
        intermediate, _ = pad_packed_sequence(intermediate, batch_first=True, padding_value=0, total_length=n_timesteps)

        for output_layer in self.output_layers:
            output = output_layer(intermediate)

            # Manually recreate Keras Masking
            # In Keras masking a mask means the last non-masked input is used
            for i in range(len(seq_lengths)):
                pad_i = seq_lengths[i]
                output[i, pad_i:, :] = output[i, pad_i - 1, :]

            if apply_activation:
                output = torch.sigmoid(output)

            outputs.append(output)

        self.h_c = h_c

        return outputs


class ComparisonFNN(nn.Module):
    def __init__(self, input_size, hidden_size=2250, output_size=1, num_targets=1):
        """
        Init of Feed Forward Neural Net class for comparison of NN vs LSTM to determine the need of time series
        @param input_size: number of input units
        @param hidden_size: number of hidden units [defaults to 2250]
        @param output_size: number of output units [defaults to 1]
        """
        super(ComparisonFNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU()
        )
        self.output_layers = nn.ModuleList([nn.Linear(hidden_size, output_size) for i in range(num_targets)])

    def forward(self, features, apply_activation=False):
        outputs = []

        intermediate = self.model(features)

        for output_layer in self.output_layers:
            output = output_layer(intermediate)
            if apply_activation:
                output = torch.sigmoid(output)
            outputs.append(output)

        return outputs


class ComparisonLogisticRegression(torch.nn.Module):
    def __init__(self, input_size, output_size=1, num_targets=1):
        super(ComparisonLogisticRegression, self).__init__()
        self.output_layers = nn.ModuleList([nn.Linear(input_size, output_size) for i in range(num_targets)])

    def forward(self, features, apply_activation=False):
        outputs = []

        for output_layer in self.output_layers:
            output = output_layer(features)
            if apply_activation:
                output = torch.sigmoid(output)
            outputs.append(output)

        return outputs
