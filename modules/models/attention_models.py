import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from modules.pad_sequences import get_seq_length_from_padded_seq


class AttentionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=256, output_size=1, num_layers=1, full_attention=True):
        """
        LSTM model which incorporates an attention mechanism
        @param input_size: number of input units
        @param hidden_size: number of hidden units [defaults to 256]
        @param output_size: number of output units [defaults to 1]
        @param num_layers: number of layers [defaults to 1]
        @param full_attention: use attention over time and features or only time for each feature [defaults to over
        time and features]
        """
        super(AttentionLSTM, self).__init__()

        self.attention_layer = nn.Linear(input_size, input_size, bias=False)
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.dense = nn.Linear(hidden_size, output_size)

        self.attention = None
        self.h_c = None
        self.full_attention = full_attention

        self.init_weights()

    def init_weights(self):
        """
        Reproduce Keras default initialization weights to initialize Embeddings/LSTM weights
        """
        ih = (param.data for name, param in self.named_parameters() if 'lstm.weight_ih' in name)
        hh = (param.data for name, param in self.named_parameters() if 'lstm.weight_hh' in name)
        b = (param.data for name, param in self.named_parameters() if 'lstm.bias' in name)
        for t in ih:
            nn.init.xavier_uniform_(t)
        for t in hh:
            nn.init.orthogonal_(t)
        for t in b:
            nn.init.constant_(t, 0)
            # Reproducing Keras' unit_forget_bias parameter
            # https://discuss.pytorch.org/t/set-forget-gate-bias-of-lstm/1745
            # It’s not super convenient, but we guarantee that a bias vector of each LSTM layer is structured like this:
            # [b_ig | b_fg | b_gg | b_og]
            n = t.size(0)
            start, end = n // 4, n // 2
            t[start:end].fill_(1.)

    def forward(self, features, h_c=None):
        n_timesteps = features.shape[1]

        # x is of shape batch_size x seq_length x n_features
        attention = self.attention_layer(features)
        if self.full_attention:
            attention = attention.reshape(features.shape[0], -1)
        attention = torch.softmax(attention, dim=1)
        if self.full_attention:
            attention = attention.reshape(features.shape[0], features.shape[1], features.shape[2])
        # Save a to attention variable for being able to return it later
        self.attention = attention.clone().detach().cpu().numpy()
        features = attention * features

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
