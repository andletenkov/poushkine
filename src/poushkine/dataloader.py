from typing import Literal

import torch

SplitType = Literal["train", "val"]


class Dataloader:
    """Provides data of given shape for train/val."""

    def __init__(
        self,
        data: list[int],
        train_size: float,
        batch_size: int,
        context_length: int,
    ) -> None:
        """Initializer.

        :param data: List of tokens.
        :param train_size: Proportion of the dataset to include in the
            train split. Must be between 0 and 1.
        :param batch_size: Num of examples to provide in single batch.
        :param context_length: Block size to keep within the context
            (time dimension).
        """
        data = torch.tensor(data, dtype=torch.long)
        split = int(len(data) * train_size)

        self._data = {"train": data[:split], "val": data[split:]}
        self._batch_size = batch_size
        self._context_length = context_length

    def get_batch(self, split: SplitType) -> tuple[torch.Tensor, torch.Tensor]:
        """Provides randomly picked batch of data examples.

        :param split: Split type. (train/val).
        :return: Tuple of feature and label tensors.
        """
        data = self._data[split]
        idx = torch.randint(len(data) - self._context_length, (self._batch_size,))
        x = torch.stack([data[i : i + self._context_length] for i in idx])
        y = torch.stack([data[i + 1 : i + self._context_length + 1] for i in idx])
        return x, y
