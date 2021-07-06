import torch
from torch import nn

from modules.hopfieldlayers import Hopfield, HopfieldPooling, HopfieldLayer


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

    def forward(self, features):
        outputs = []

        intermediate = self.hopfield(features)

        for output_layer in self.output_layers:
            output = torch.sigmoid(output_layer(intermediate))
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

    def forward(self, features):
        outputs = []
        intermediate = self.hopfield_pooling(features)
        intermediate = intermediate.reshape(features.shape[0], features.shape[1], features.shape[2])

        for output_layer in self.output_layers:
            output = torch.sigmoid(output_layer(intermediate))
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

    def forward(self, features):
        outputs = []

        intermediate = self.hopfield_lookup(features)

        for output_layer in self.output_layers:
            output = torch.sigmoid(output_layer(intermediate))
            outputs.append(output)

        return outputs
