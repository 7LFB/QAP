# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import math

from typing import List, Type


class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


# From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
# Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
    

class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        mlp_dims: List[int],
        dropout: float = 0.1,
        nonlinearity: Type[nn.Module] = nn.ReLU,
        normalization: Type[nn.Module] = nn.BatchNorm1d,  # nn.LayerNorm,
        special_bias: bool = False,
        add_bn_first: bool = False,
    ):
        super(MLP, self).__init__()
        projection_prev_dim = input_dim
        projection_modulelist = []
        last_dim = mlp_dims[-1]
        mlp_dims = mlp_dims[:-1]

        if add_bn_first:
            if normalization is not None:
                projection_modulelist.append(normalization(projection_prev_dim))
            if dropout != 0:
                projection_modulelist.append(nn.Dropout(dropout))

        for idx, mlp_dim in enumerate(mlp_dims):
            fc_layer = nn.Linear(projection_prev_dim, mlp_dim)
            nn.init.kaiming_normal_(fc_layer.weight, a=0, mode='fan_out')
            projection_modulelist.append(fc_layer)
            projection_modulelist.append(nonlinearity())

            if normalization is not None:
                projection_modulelist.append(normalization(mlp_dim))

            if dropout != 0:
                projection_modulelist.append(nn.Dropout(dropout))
            projection_prev_dim = mlp_dim

        self.projection = nn.Sequential(*projection_modulelist)
        self.last_layer = nn.Linear(projection_prev_dim, last_dim)
        nn.init.kaiming_normal_(self.last_layer.weight, a=0, mode='fan_out')
        if special_bias:
            prior_prob = 0.01
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            torch.nn.init.constant_(self.last_layer.bias, bias_value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        input_arguments:
            @x: torch.FloatTensor
        """
        x = self.projection(x)
        x = self.last_layer(x)
        return x
    
    

