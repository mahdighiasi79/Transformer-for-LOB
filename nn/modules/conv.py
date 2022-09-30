from copy import deepcopy
from typing import List
from typing import Tuple
from typing import Union

from torch import Tensor
from torch.nn import Conv1d
from torch.nn import Module
from torch.nn import ReLU
from torch.nn import ReplicationPad1d
from torch.nn import Sequential


class CausalConv1d(Sequential):

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1
    ):
        super().__init__()
        self.pad = ReplicationPad1d(((kernel_size - 1) * dilation, 0))
        self.conv = Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)

    def forward(self, input: Tensor) -> Tensor:
        return self.conv(self.pad(input))

    def __repr__(self) -> str:
        return self._get_name() + f"({self.conv.extra_repr()})"


class CausalConvLayers(Sequential):

    def __init__(
        self,
        in_channels: int,
        n_features: int,
        kernel_size: int,
        dilation: Union[Tuple[int, ...], int] = 1,
        n_layers: int = 5,
        activation: Module = ReLU(),
    ):
        if isinstance(dilation, int):
            dilation = (dilation,) * n_layers

        layers: List[Module] = []
        for i in range(n_layers):
            c = in_channels if i == 0 else n_features
            layers.append(CausalConv1d(c, n_features, kernel_size, dilation[i]))
            if i != n_layers - 1:
                layers.append(deepcopy(activation))

        super().__init__(*layers)
