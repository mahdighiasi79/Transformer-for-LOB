from typing import Tuple
from typing import Union

import torch
from torch import Tensor
from torch.nn import Dropout
from torch.nn import Flatten
from torch.nn import LayerNorm
from torch.nn import Linear
from torch.nn import Module
from torch.nn import Sequential
from torch.nn import Softmax
from torch.nn import TransformerEncoder

from .conv import CausalConvLayers
from .mlp import MultiLayerPerceptron
from .position import PositionalEncoding
from .transformer import CausalTransformerEncoderLayer


class TransLOB(Module):

    def __init__(
        self,
        in_features: int = 40,
        out_features: int = 3,
        len_sequence: int = 100,
        out_activation: Module = Softmax(-1),
        conv_n_features: int = 14,
        conv_kernel_size: int = 2,
        conv_dilation: Union[Tuple[int, ...], int] = (1, 2, 4, 8, 16),
        conv_n_layers: int = 5,
        tf_n_channels: int = 3,
        tf_dim_feedforward: int = 60,
        tf_dropout_rate: float = 0.0,
        tf_num_layers: int = 2,
        mlp_dim: int = 64,
        mlp_n_layers: int = 1,
        dropout_rate: float = 0.1,
    ):
        super().__init__()

        # Define convolutional module.
        convolution = CausalConvLayers(
            in_features,
            conv_n_features,
            conv_kernel_size,
            dilation=conv_dilation,
            n_layers=conv_n_layers,
        )
        self.pre_transformer = Sequential(
            convolution,
            LayerNorm(torch.Size((conv_n_features, len_sequence))),
            PositionalEncoding(max_length=len_sequence),
        )

        # Define Transformer encoder module.
        d_model = conv_n_features + 1
        encoder_layer = CausalTransformerEncoderLayer(
            d_model=d_model,
            nhead=tf_n_channels,
            dim_feedforward=tf_dim_feedforward,
            dropout=tf_dropout_rate,
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers=tf_num_layers)

        # Define modules used after Transformer encoder.
        multi_layer_perceptron = MultiLayerPerceptron(
            in_features=d_model * len_sequence,
            out_features=mlp_dim,
            n_layers=mlp_n_layers,
            n_units=mlp_dim,
        )
        self.post_transformer = Sequential(
            Flatten(1, -1),
            multi_layer_perceptron,
            Dropout(dropout_rate),
            Linear(mlp_dim, out_features),
            out_activation,
        )

    def forward(self, input: Tensor) -> Tensor:
        input = self.pre_transformer(input).movedim(-1, 0)
        input = self.transformer(input)
        input = self.post_transformer(input.movedim(0, -1))
        return input
