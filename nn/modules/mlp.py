from copy import deepcopy
from typing import List

from torch.nn import Linear
from torch.nn import Module
from torch.nn import ReLU
from torch.nn import Sequential


class MultiLayerPerceptron(Sequential):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_layers: int = 2,
        n_units: int = 32,
        activation: Module = ReLU(),
    ) -> None:
        layers: List[Module] = []
        for i_layer in range(n_layers):
            layers.append(Linear(in_features if i_layer == 0 else n_units, n_units))
            layers.append(deepcopy(activation))
        layers.append(Linear(n_units, out_features))

        super().__init__(*layers)
