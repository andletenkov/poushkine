import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class BigramModel(nn.Module):
    """Simple language model predicting next most probable character."""

    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        num_embed: int,
        device: torch.device,
        *args,
        **kwargs
    ) -> None:
        """Initializer.

        Args:
            vocab_size (int):  Number of unique tokens to embed (0 dim).
            block_size (int): Block size (context length) to embed token positions (0 dim).
            num_embed (int): Number of embeddings for a single char (1 dim).
            device (torch.device): Device to which model and its parameters should be moved during the training (cpu/cuda).
        """
        super().__init__(*args, **kwargs)
        self.device = device
        self._block_size = block_size
        self._token_embedding = nn.Embedding(vocab_size, num_embed)
        self._position_embedding = nn.Embedding(block_size, num_embed)
        self._head = nn.Linear(num_embed, vocab_size)

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
        head_input = token_emb + pos_emb
        logits = self._head(
            head_input
        )  # shape: (batch size, context length, vocab size)

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
