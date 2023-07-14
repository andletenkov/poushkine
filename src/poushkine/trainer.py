from typing import Type, get_args

import torch
import torch.optim as optim
from tqdm import tqdm

from poushkine.dataloader import Dataloader, SplitType
from poushkine.model import BigramModel


class Trainer:
    """Custom model trainer class."""

    def __init__(
        self,
        model: BigramModel,
        dataloader: Dataloader,
        optimizer_cls: Type[optim.Optimizer],
        max_iterations: int = 10000,
        eval_iterations: int = 200,
        eval_interval: int = 100,
        **optimizer_params
    ) -> None:
        """Initializer.

        Args:
            model (nn.Module): A custom bigram model to train.
            dataloader (Dataloader): Dataloader instance.
            optimizer_cls (Type[optim.Optimizer]): Optimizer class.
            learning_rate (float, optional): Learning rate to use for the optimizer. Defaults to 1e-3.
            max_iterations (int, optional): Number of optimizer steps to perform. Defaults to 10000.
            eval_iterations (int, optional): Number of model evaluations to estimate train/val loss. Defaults to 200.
            eval_interval (int, optional): Number of steps between every loss estimation. Defaults to 100.
            optimizer_params: Any keyword arguments for the optimizer.
        """
        self._device = model.device
        self._model = model.to(self._device)
        self._dataloader = dataloader
        self._optimizer = optimizer_cls(self._model.parameters(), **optimizer_params)
        self._max_iterations = max_iterations
        self._eval_iterations = eval_iterations
        self._eval_interval = eval_interval

    @torch.no_grad()
    def _estimate_loss(self) -> dict[SplitType, float]:
        """Esimates mean training and validations losses.

        Returns:
            _type_: Dictionary with calculated loss values for every kind of data split (train/val).
        """
        loss_dict = {}
        self._model.eval()
        try:
            for split in get_args(SplitType):
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
