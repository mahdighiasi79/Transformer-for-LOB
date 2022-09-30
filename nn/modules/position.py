import math

import torch
from torch import Tensor
from torch.nn import Module


class PositionalEncoding(Module):

    positional_encoder: Tensor

    def __init__(
        self, d_model: int = 1, max_length: int = 100, encoding: str = "linear"
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.max_length = max_length
        self.encoding = encoding

        self.register_buffer("positional_encoder", self._compute_positional_encoder())

    def _compute_positional_encoder(self) -> Tensor:
        # Returns:
        # positional_encoder : tensor, shape (F, L)
        #     F : number of features
        #     L : maximum length
        if self.encoding == "sinusoid":
            position = torch.linspace(0.0, 2 * math.pi, self.max_length).reshape(-1, 1)
            frequency = torch.logspace(0.0, math.log(2 * math.pi), self.d_model, math.e)
            frequency = frequency.unsqueeze(0)

            phase = frequency * position

            positional_encoder = torch.empty((self.max_length, 2 * self.d_model))
            positional_encoder[:, 0::2] = phase.sin()
            positional_encoder[:, 1::2] = phase.cos()

        if self.encoding == "linear":
            positional_encoder = torch.linspace(0.0, 1.0, self.max_length).unsqueeze(0)
        else:
            raise ValueError("invalid 'encoding'")

        return positional_encoder

    def forward(self, input: Tensor) -> Tensor:
        # cut and align shape
        p = self.positional_encoder[..., : input.size(-1)]
        # for input shape (N, *, X, L), p's shape is (N, *, F, L)
        p = p.expand(input.size()[:-2] + p.size()[-2:])
        return torch.cat((input, p), -2)
