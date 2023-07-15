import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttentionHead(nn.Module):
    """Self-attention head implementation."""

    def __init__(
        self,
        num_embed: int,
        head_size: int,
        block_size: int,
        dropout: float,
        *args,
        **kwargs
    ) -> None:
        """initializer.

        Args:
            num_embed (int): Number of input token embeddings.
            block_size (int): Self-attention context length.
            head_size (int): Self-attention head size.
            dropout (float): Probability to zero an element of an input tensor.
        """
        super().__init__(*args, **kwargs)
        self.head_size = head_size
        self._query = nn.Linear(num_embed, head_size, bias=False)
        self._key = nn.Linear(num_embed, head_size, bias=False)
        self._value = nn.Linear(num_embed, head_size, bias=False)
        self._dropout = nn.Dropout(dropout)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Self-attention forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch size, block size, num embeddings).

        Returns:
            torch.Tensor: Output tensor of shape (batch size, block size, head size).
        """
        _, x_block_size, _ = x.shape

        q = self._query(x)  # (batch_size, block_size, head_size)
        k = self._key(x)  # (batch_size, block_size, head_size)
        v = self._value(x)  # (batch_size, block_size, head_size)

        affinities = (
            q @ k.transpose(-2, -1) * self.head_size**-0.5
        )  # (batch_size, block_size, block_size)
        tril: torch.Tensor = getattr(self, "tril")
        affinities = affinities.masked_fill(
            tril[:x_block_size, :x_block_size] == 0, float("-inf")
        )
        affinities = F.softmax(affinities, dim=-1)
        affinities = self._dropout(affinities)

        out = affinities @ v  # (batch_size, block_size, head_size)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        num_embed: int,
        block_size: int,
        dropout: float,
        *args,
        **kwargs
    ) -> None:
        """Initializer.

        Args:
            num_heads (int): Number of self-attention heads to use in parallel.
            block_size (int): Self-attention context length.
            num_embed (int): Number of token embeddings (self-attention head size).
            dropout (float): Probability to zero an element of an input tensor.
        """
        super().__init__(*args, **kwargs)
        single_head_size = num_embed // num_heads
        self._heads = nn.ModuleList(
            [
                SelfAttentionHead(
                    num_embed=num_embed,
                    head_size=single_head_size,
                    block_size=block_size,
                    dropout=dropout,
                )
                for _ in range(num_heads)
            ]
        )
        self._proj = nn.Linear(num_embed, num_embed)
        self._dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Multi-head attention forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch size, block size, num embeddings).

            torch.Tensor: Output tensor of shape (batch size, block size, head size).
        """
        x = torch.cat([h(x) for h in self._heads], dim=-1)
        x = self._proj(x)
        out = self._dropout(x)
        return out
