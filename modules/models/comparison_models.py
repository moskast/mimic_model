import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from modules.pad_sequences import get_seq_length_from_padded_seq


class ComparisonLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=256, output_size=1, num_layers=1):
        """
        Init of LSTM class for comparison of NN vs LSTM to determine the need of time series
        @param input_size: number of input units
        @param hidden_size: number of hidden units [defaults to 256]
        @param output_size: number of output units [defaults to 1]
        @param num_layers: number of layers [defaults to 1]
        """
        super(ComparisonLSTM, self).__init__()

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.dense = nn.Linear(hidden_size, output_size)

        self.h_c = None

    def forward(self, features, h_c=None):
        n_timesteps = features.shape[1]

        seq_lengths = get_seq_length_from_padded_seq(features.clone().detach().cpu().numpy())
        features = pack_padded_sequence(features, seq_lengths, batch_first=True, enforce_sorted=False)
        if h_c is None:
            intermediate, h_c = self.lstm(features)
        else:
            h, c = h_c
            intermediate, h_c = self.lstm(features, h, c)
        intermediate, _ = pad_packed_sequence(intermediate, batch_first=True, padding_value=0, total_length=n_timesteps)

        intermediate = self.dense(intermediate)

        # Manually recreate Keras Masking
        # In Keras masking a mask means the last non-masked input is used
        for i in range(len(seq_lengths)):
            pad_i = seq_lengths[i]
            intermediate[i, pad_i:, :] = intermediate[i, pad_i - 1, :]

        output = torch.sigmoid(intermediate)

        self.h_c = h_c

        return output


class ComparisonFFNN(nn.Module):
    def __init__(self, input_size, hidden_size=2250,  output_size=1):
        """
        Init of Feed Forward Neural Net class for comparison of NN vs LSTM to determine the need of time series
        @param input_size: number of input units
        @param hidden_size: number of hidden units [defaults to 2250]
        @param output_size: number of output units [defaults to 1]
        """
        super(ComparisonFFNN, self).__init__()
        self.linear = nn.Linear(input_size, hidden_size)
        self.dense = nn.Linear(hidden_size, output_size)

        self.attention = None

    def forward(self, features):
        intermediate = self.linear(features)
        intermediate = self.dense(intermediate)
        output = torch.sigmoid(intermediate)

        return output
