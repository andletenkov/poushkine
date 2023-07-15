import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from poushkine.attention import MultiHeadAttention


class FeedForward(nn.Module):
    """A simple linear layer followed by non-linear activation."""

    def __init__(self, num_embed: int, dropout: float, *args, **kwargs) -> None:
        """Initializer.

        Args:
            num_embed (int): Number of token embeddings.
            dropout (float): Probability to zero an element of an input tensor.
        """
        super().__init__(*args, **kwargs)
        self._ff = nn.Sequential(
            nn.Linear(num_embed, num_embed * 4),
            nn.ReLU(),
            nn.Linear(num_embed * 4, num_embed),  # linear projection
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Feed forward pass.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        return self._ff(x)


class TransformerBlock(nn.Module):
    """A transformer block. Communication + computation."""

    def __init__(
        self,
        num_embed: int,
        block_size: int,
        attention_num_heads: int,
        dropout: float,
        *args,
        **kwargs
    ) -> None:
        """Initializer.

        Args:
            num_embed (int): Number of token embeddings.
            block_size (int): Tokens block size (context length).
            attention_num_heads (int): Number of self-attention heads to use in parallel.
            dropout (float): Probability to zero an element of an input tensor.
        """
        super().__init__(*args, **kwargs)

        self._attention = MultiHeadAttention(
            attention_num_heads, num_embed, block_size, dropout
        )
        self._ff = FeedForward(num_embed, dropout)
        self._ln1 = nn.LayerNorm(num_embed)
        self._ln2 = nn.LayerNorm(num_embed)

    def forward(self, x: Tensor) -> Tensor:
        """Block forward pass.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        # use residual connections
        x = x + self._attention(self._ln1(x))
        x = x + self._ff(self._ln2(x))
        return x


class BigramModel(nn.Module):
    """Simple language model predicting next most probable character."""

    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        num_embed: int,
        attention_num_heads: int,
        dropout: float,
        num_blocks: int,
        device: torch.device,
        *args,
        **kwargs
    ) -> None:
        """Initializer.

        Args:
            vocab_size (int):  Number of unique tokens the model is able to handle.
            block_size (int): Tokens block size (context length).
            num_embed (int): Number of embeddings for a single char.
            attention_num_heads (int). Number of self-attention heads to use in parallel.
            dropout (float): Probability to zero an element of an input tensor.
            num_blocks (int): Number of transformer blocks.
            device (torch.device): Device to which model and its parameters should be moved
                during the training (cpu/cuda).
        """
        super().__init__(*args, **kwargs)
        self.device = device
        self._block_size = block_size
        self._token_embedding = nn.Embedding(vocab_size, num_embed)
        self._position_embedding = nn.Embedding(block_size, num_embed)
        self._head = nn.Linear(num_embed, vocab_size)
        self._blocks = nn.Sequential(
            *[
                TransformerBlock(num_embed, block_size, attention_num_heads, dropout)
                for _ in range(num_blocks)
            ],
            nn.LayerNorm(num_embed),
        )

    def forward(
        self, idx: Tensor, targets: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
        """Forward pass.

        :param idx: Features tensor of shape (batch size, context
            length)
        :param targets: Targets tensor of shape (batch size, context
            length)
        :return: Tuple of logits and loss tensors.
        """
        _, ctx_len = idx.shape

        pos_emb = self._position_embedding(
            torch.arange(ctx_len, device=self.device)
        )  # shape: (context length, num embed)
        token_emb = self._token_embedding(
            idx
        )  # shape: (batch size, context length, num embed)
        x = token_emb + pos_emb
        x = self._blocks(x)
        logits = self._head(x)  # shape: (batch size, context length, vocab size)

        if targets is None:
            return logits, None

        batch_dim, ctx_dim, class_dim = logits.shape
        logits = logits.view(batch_dim * ctx_dim, class_dim)
        targets = targets.view(batch_dim * ctx_dim)

        loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, num_tokens: int, start: Tensor | None = None) -> Tensor:
        """Generates new tokens based on given context.

        Args:
            num_tokens (int): Number of new tokens to generate.
            start (Tensor, optional): Tensor with initial context (first dimention should be equal to 1).
                Defaults to [[0]] tensor.

        Returns:
            Tensor: One-dimentional tensor with generated tokens.
        """
        idx = (
            torch.zeros((1, 1), dtype=torch.long, device=self.device)
            if start is None
            else start
        )
        assert idx.shape[0] == 1

        for _ in range(num_tokens):
            # consider only block_size last tokens
            idx_cropped = idx[:, -self._block_size :]
            logits, _ = self(idx_cropped)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx[0]
