from typing import Any, Dict, List, Mapping, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


class CondSequential(torch.nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.m = torch.nn.ModuleList(args)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        for m in self.m:
            x = m(x, y)
        return x


class ResBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        bias: bool = False,
        num_cond_features: int = 0,
        dim: int = 3,
    ):
        super().__init__()

        conv = torch.nn.Conv3d if dim == 3 else torch.nn.Conv2d

        self.c1 = conv(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, bias=bias
        )
        if num_cond_features:
            self.affine = ConditionalAffine(
                num_features=out_channels, num_cond_features=num_cond_features, bias=bias, dim=dim
            )
        else:
            self.affine = Identity()
        self.act = torch.nn.SiLU()
        self.c2 = conv(
            in_channels=out_channels, out_channels=in_channels, kernel_size=kernel_size, padding=padding, bias=bias
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        h = self.c1(x)
        h = self.affine(h, y)
        h = self.act(h)
        h = self.c2(h)
        return x + h


class DownBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mode: str = "conv",
        kernel_size: int = 3,
        stride: int = 2,
        bias: bool = True,
        dim: int = 3,
    ):
        super().__init__()

        if mode == "conv":
            conv = torch.nn.Conv3d if dim == 3 else torch.nn.Conv2d
            self.m = conv(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                bias=bias,
            )
        elif mode == "avgpool":
            pool = torch.nn.AvgPool3d if dim == 3 else torch.nn.AvgPool2d
            self.m = pool(
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
            )
        elif mode == "maxpool":
            pool = torch.nn.MaxPool3d if dim == 3 else torch.nn.MaxPool2d
            self.m = pool(
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
            )
        else:
            raise RuntimeError("unkown mode")

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.m(x)


class UpBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mode: str = "conv",
        kernel_size: int = 3,
        stride: int = 2,
        bias: bool = True,
        dim: int = 3,
    ):
        super().__init__()

        if mode == "conv":
            conv = torch.nn.ConvTranspose3d if dim == 3 else torch.nn.ConvTranspose2d
            self.m = conv(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                bias=bias,
            )
        elif mode == "interpolate":
            pass
            # TODO!
        else:
            raise RuntimeError("unkown mode")

    def forward(self, x: torch.Tensor, shape: tuple) -> torch.Tensor:
        return self.m(x, output_size=shape)


class Identity(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x


class ConditionalAffine(torch.nn.Module):
    def __init__(self, num_features: int, num_cond_features: int, bias: bool = False, dim: int = 3):
        super().__init__()

        self.num_features = num_features
        self.bias = bias
        self.dim = dim
        self.linear = torch.nn.Linear(in_features=num_cond_features, out_features=(2 if bias else 1) * num_features)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.bias:
            gamma, beta = self.linear(y).chunk(2, dim=1)
        else:
            gamma = self.linear(y)
            beta = torch.zeros_like(gamma)

        gamma = F.softplus(gamma)

        if self.dim == 3:
            return gamma.view(-1, self.num_features, 1, 1, 1) * x + beta.view(-1, self.num_features, 1, 1, 1)
        else:
            return gamma.view(-1, self.num_features, 1, 1) * x + beta.view(-1, self.num_features, 1, 1)


class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""

    def __init__(self, embed_dim, scale=30.0):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class FeedForward(torch.nn.Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: int) -> None:
        super().__init__()

        self.net = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=in_features,
                out_features=hidden_features,
            ),
            torch.nn.SiLU(),
            torch.nn.Linear(in_features=hidden_features, out_features=out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DRUNet(torch.nn.Module):
    # https://github.com/cszn/DPIR/blob/master/models/network_unet.py
    def __init__(
        self,
        dim: int,  # 2, or 3 dimensional
        nc: List[int],
        nb: int,
        bias: bool,
        num_cond_features: int,
        in_channels: int,
        out_channels: int,
        label_dim: int,  # Number of class labels, 0 = unconditional
        label_dropout: float,  # Dropout probability of class labels for classifier-free guidance
        embedding: Dict[str, Any],
    ):
        super().__init__()

        assert dim in {2, 3}, "dim must be either 2 or 3"
        self.dim = dim

        self.t_embedder = FeedForward(**embedding)
        self.y_embedder = LabelEmbedder(label_dim, num_cond_features, label_dropout) if label_dim else None
        # self.embedding = nn.Sequential(
        #     GaussianFourierProjection(embed_dim=num_cond_features),
        #     nn.Linear(num_cond_features, num_cond_features),
        #     nn.SiLU()
        # )

        conv = torch.nn.Conv3d if dim == 3 else torch.nn.Conv2d

        self.m_head = conv(in_channels=in_channels, out_channels=nc[0], kernel_size=3, padding=1, bias=bias)

        # TODO: write as Modulelist for number of scales
        self.m_enc1 = CondSequential(
            *[ResBlock(nc[0], nc[0], bias=bias, num_cond_features=num_cond_features, dim=dim) for _ in range(nb)]
        )
        self.m_enc2 = CondSequential(
            *[ResBlock(nc[1], nc[1], bias=bias, num_cond_features=num_cond_features, dim=dim) for _ in range(nb)]
        )
        self.m_enc3 = CondSequential(
            *[ResBlock(nc[2], nc[2], bias=bias, num_cond_features=num_cond_features, dim=dim) for _ in range(nb)]
        )
        self.m_down = torch.nn.ModuleList(
            [DownBlock(nc[i], nc[i + 1], bias=bias, mode="conv", dim=dim) for i in range(3)]
        )

        self.m_body = CondSequential(
            *[ResBlock(nc[3], nc[3], bias=bias, num_cond_features=num_cond_features, dim=dim) for _ in range(nb)]
        )

        self.m_up = torch.nn.ModuleList([UpBlock(nc[i + 1], nc[i], bias=bias, mode="conv", dim=dim) for i in range(3)])
        self.m_dec3 = CondSequential(
            *[ResBlock(nc[2], nc[2], bias=bias, num_cond_features=num_cond_features, dim=dim) for _ in range(nb)]
        )
        self.m_dec2 = CondSequential(
            *[ResBlock(nc[1], nc[1], bias=bias, num_cond_features=num_cond_features, dim=dim) for _ in range(nb)]
        )
        self.m_dec1 = CondSequential(
            *[ResBlock(nc[0], nc[0], bias=bias, num_cond_features=num_cond_features, dim=dim) for _ in range(nb)]
        )

        self.m_tail = conv(in_channels=nc[0], out_channels=out_channels, kernel_size=3, padding=1, bias=bias)

    def forward(
        self, x: torch.Tensor, noise_labels: torch.Tensor, class_labels: torch.Tensor, augment_labels=None
    ) -> torch.Tensor:
        emb = self.t_embedder(noise_labels.view(x.shape[0], -1))  # (B, D)

        if self.y_embedder is not None:
            class_labels = self.y_embedder(class_labels.flatten(), self.training)  # (B, D)
            emb = emb + class_labels

        x1 = self.m_head(x)
        x1 = self.m_enc1(x1, emb)
        x2 = self.m_down[0](x1)
        x2 = self.m_enc2(x2, emb)
        x3 = self.m_down[1](x2)
        x3 = self.m_enc3(x3, emb)
        x4 = self.m_down[2](x3)
        x = self.m_body(x4, emb)
        x = self.m_up[2](x + x4, x3.shape)
        x = self.m_dec3(x, emb)
        x = self.m_up[1](x + x3, x2.shape)
        x = self.m_dec2(x, emb)
        x = self.m_up[0](x + x2, x1.shape)
        x = self.m_dec1(x, emb)
        x = self.m_tail(x + x1)

        return x


if __name__ == "__main__":
    unet = DRUNet(
        dim=3,
        nc=[32, 64, 128, 256],
        nb=2,
        bias=False,
        num_cond_features=64,
        in_channels=1,
        out_channels=1,
        label_dim=0,
        label_dropout=0.0,
        embedding=dict(in_features=1, hidden_features=64, out_features=64),
    )

    print(unet)
