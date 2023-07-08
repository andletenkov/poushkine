from typing import Literal, Type

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from poushkine.dataloader import Dataloader


class Trainer:
    """Custom model trainer class."""

    def __init__(
        self,
        model: nn.Module,
        dataloader: Dataloader,
        optimizer_cls: Type[optim.Optimizer],
        device: torch.device = torch.device("cpu"),
        learning_rate: float = 1e-3,
        max_iterations: int = 10000,
        eval_iterations: int = 200,
        eval_interval: int = 100,
    ) -> None:
        """Initializer.

        Args:
            model (nn.Module): A model to train.
            dataloader (Dataloader): Dataloader instance.
            optimizer_cls (Type[optim.Optimizer]): Optimizer class.
            device (torch.device, optional): Torch device to which move the model and the data. Defaults to torch.device("cpu").
            learning_rate (float, optional): Learning rate to use for the optimizer. Defaults to 1e-3.
            max_iterations (int, optional): Number of optimizer steps to perform. Defaults to 10000.
            eval_iterations (int, optional): Number of model evaluations to estimate train/val loss. Defaults to 200.
            eval_interval (int, optional): Number of steps between every loss estimation. Defaults to 100.
        """
        self._device = device
        self._model = model.to(self._device)
        self._dataloader = dataloader
        self._optimizer = optimizer_cls(self._model.parameters(), lr=learning_rate)
        self._max_iterations = max_iterations
        self._eval_iterations = eval_iterations
        self._eval_interval = eval_interval

    @torch.no_grad()
    def _estimate_loss(self) -> dict[Literal["train", "val"], float]:
        """Esimates mean training and validations losses.

        Returns:
            _type_: Dictionary with calculated loss values for every kind of data split (train/val).
        """
        loss_dict = {}
        self._model.eval()
        try:
            for split in ["train", "val"]:
                losses = torch.zeros(self._eval_iterations)
                for i in range(self._eval_iterations):
                    x, y = self._dataloader.get_batch(split, self._device)
                    _, loss = self._model(x, y)
                    losses[i] = loss
                loss_dict[split] = losses.mean().item()
        finally:
            self._model.train()
        return loss_dict

    def train(self):
        """Model trainig loop."""
        iters = range(self._max_iterations)
        with tqdm(iters) as pbar:
            for i in iters:
                x_batch, y_batch = self._dataloader.get_batch("train", self._device)
                _, loss = self._model(x_batch, y_batch)

                self._optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self._optimizer.step()

                if i % self._eval_interval == 0:
                    split_losses = self._estimate_loss()
                    pbar.set_postfix(
                        iter=i,
                        train_loss=split_losses["train"],
                        val_loss=split_losses["val"],
                    )

                pbar.update(1)
