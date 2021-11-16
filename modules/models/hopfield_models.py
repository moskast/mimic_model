import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from modules.config import AppConfig
from modules.hopfieldlayers import Hopfield, HopfieldPooling, HopfieldLayer
from modules.utils.pad_sequences import get_seq_length_from_padded_seq


class HopfieldLayerModel(nn.Module):
    def __init__(self, input_size, hidden_size=617, output_size=1, num_targets=1):
        """

        @param input_size:
        @param hidden_size:
        @param output_size:
        """
        super(HopfieldLayerModel, self).__init__()
        self.hopfield = Hopfield(input_size=input_size, hidden_size=hidden_size)

        self.output_layers = nn.ModuleList(
            [nn.Linear(self.hopfield.output_size, output_size) for i in range(num_targets)])

    def forward(self, features, apply_activation=False, device=AppConfig.device):
        outputs = []
        association_mask = torch.triu(
            torch.ones((features.shape[1], features.shape[1])), diagonal=1
        ).to(device)
        association_mask = association_mask.type(torch.bool)
        intermediate = self.hopfield(features, association_mask=association_mask)

        for output_layer in self.output_layers:
            output = output_layer(intermediate)
            if apply_activation:
                output = torch.sigmoid(output)
            outputs.append(output)

        return outputs


class HopfieldPoolingModel(nn.Module):
    def __init__(self, input_size, hidden_size=614, output_size=1, num_targets=1):
        """

        @param input_size:
        @param hidden_size:
        @param output_size:
        """
        super(HopfieldPoolingModel, self).__init__()
        self.hopfield_pooling = HopfieldPooling(input_size=input_size, hidden_size=hidden_size, quantity=13)

        self.output_layers = nn.ModuleList(
            [nn.Linear(self.hopfield_pooling.output_size, output_size) for i in range(num_targets)])

    def forward(self, features, apply_activation=False):
        outputs = []
        intermediate = self.hopfield_pooling(features)
        intermediate = intermediate.reshape(features.shape[0], features.shape[1], features.shape[2])

        for output_layer in self.output_layers:
            output = output_layer(intermediate)
            if apply_activation:
                output = torch.sigmoid(output)
            outputs.append(output)

        return outputs


class HopfieldLookupModel(nn.Module):
    def __init__(self, input_size, quantity=1625, output_size=1, num_targets=1):
        """

        @param input_size:
        @param quantity:
        @param output_size:
        """
        super(HopfieldLookupModel, self).__init__()
        if quantity <= 0:
            quantity = 1
        self.hopfield_lookup = HopfieldLayer(input_size=input_size, quantity=quantity)

        self.output_layers = nn.ModuleList(
            [nn.Linear(self.hopfield_lookup.output_size, output_size) for i in range(num_targets)])

    def forward(self, features, apply_activation=False):
        outputs = []
        intermediate = self.hopfield_lookup(features)

        for output_layer in self.output_layers:
            output = output_layer(intermediate)
            if apply_activation:
                output = torch.sigmoid(output)
            outputs.append(output)

        return outputs


class HopfieldLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=256, output_size=1, num_layers=1, num_targets=1, full_attention=True):
        """
        LSTM model which incorporates an attention mechanism
        @param input_size: number of input units
        @param hidden_size: number of hidden units [defaults to 256]
        @param output_size: number of output units [defaults to 1]
        @param num_layers: number of layers [defaults to 1]
        @param full_attention: use attention over time and features or only time for each feature [defaults to over
        time and features]
        """
        super(HopfieldLSTM, self).__init__()

        self.attention_layer = Hopfield(input_size=input_size, hidden_size=input_size)
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

        self.output_layers = nn.ModuleList([nn.Linear(hidden_size, output_size) for i in range(num_targets)])

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

    def forward(self, features, h_c=None, apply_activation=False, device=AppConfig.device):
        n_timesteps = features.shape[1]
        seq_lengths = get_seq_length_from_padded_seq(features.clone().detach().cpu().numpy())
        outputs = []
        association_mask = torch.triu(
            torch.ones((features.shape[1], features.shape[1])), diagonal=1
        ).to(device)
        association_mask = association_mask.type(torch.bool)
        # x is of shape batch_size x seq_length x n_features
        attention = self.attention_layer(features, association_mask=association_mask)
        if self.full_attention:
            attention = attention.reshape(features.shape[0], -1)
        attention = torch.softmax(attention, dim=1)
        if self.full_attention:
            attention = attention.reshape(features.shape[0], features.shape[1], features.shape[2])
        # Save a to attention variable for being able to return it later
        self.attention = attention.clone().detach().cpu().numpy()
        features = attention * features

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
