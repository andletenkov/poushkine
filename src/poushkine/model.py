import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class BigramModel(nn.Module):
    """Simple language model predicting next most probable character."""

    def __init__(self, vocab_size: int, *args, **kwargs) -> None:
        """Initializer.

        :param vocab_size: Total number of unique characters model can
            handle.
        :param args: nn.Module args.
        :param kwargs: nn.Module kwargs.
        """
        super().__init__(*args, **kwargs)
        self.token_embedding = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx: Tensor, targets: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        """Forward pass.

        :param idx: Features tensor of shape (batch size, context
            length)
        :param targets: Targets tensor of shape (batch size, context
            length)
        :return: Tuple of logits and loss tensors.
        """
        logits = self.token_embedding(idx)

        if targets is None:
            return logits, None

        batch_dim, ctx_dim, class_dim = logits.shape
        logits = logits.view(batch_dim * ctx_dim, class_dim)
        targets = targets.view(batch_dim * ctx_dim)

        loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, num_tokens: int) -> Tensor:
        """Generates new tokens based on given context.

        :param idx: Tensor of shape (batch size, context length)
        :param num_tokens: Number of new tokens to generate.
        :return: Tensor of shape (batch size, context length +
            num_tokens)
        """
        for _ in range(num_tokens):
            logits, loss = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx
