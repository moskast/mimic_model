from torch import nn

from modules.hopfieldlayers import Hopfield, HopfieldPooling, HopfieldLayer


class HopfieldLayerModel(nn.Module):
    def __init__(self, input_size, hidden_size=256, output_size=1):
        """

        @param input_size:
        @param hidden_size:
        @param output_size:
        """
        super(HopfieldLayerModel, self).__init__()
        self.hopfield = Hopfield(input_size=input_size, hidden_size=hidden_size)
        self.output = nn.Linear(self.hopfield.output_size * input_size, output_size)

    def forward(self, features):
        intermediate = self.hopfield(features)
        output = self.output(intermediate)
        return output


class HopfieldPoolingModel(nn.Module):
    def __init__(self, input_size, hidden_size=256, output_size=1):
        """

        @param input_size:
        @param hidden_size:
        @param output_size:
        """
        super(HopfieldPoolingModel, self).__init__()
        self.hopfield_pooling = HopfieldPooling(input_size=input_size, hidden_size=hidden_size)
        self.output = nn.Linear(self.hopfield_pooling.output_size * input_size, output_size)

    def forward(self, features):
        intermediate = self.hopfield_pooling(features)
        output = self.output(intermediate)
        return output


class HopfieldLookupModel(nn.Module):
    def __init__(self, input_size, quantity=10, output_size=1):
        """

        @param input_size:
        @param quantity:
        @param output_size:
        """
        super(HopfieldLookupModel, self).__init__()
        self.hopfield_lookup = HopfieldLayer(input_size=input_size, quantity=quantity)
        self.output = nn.Linear(self.hopfield_lookup.output_size * input_size, output_size)

    def forward(self, features):
        intermediate = self.hopfield_lookup(features)
        output = self.output(intermediate)
        return output
