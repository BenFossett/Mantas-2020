import time
import numpy as np

import torch
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from pathlib import Path
from typing import Union

def compute_accuracy(
    labels: Union[torch.Tensor, np.ndarray], preds: Union[torch.Tensor, np.ndarray]
) -> float:
    """
    Args:
        labels: ``(batch_size, class_count)`` tensor or array containing example labels
        preds: ``(batch_size, class_count)`` tensor or array containing model prediction
    """
    assert len(labels) == len(preds)
    return float((labels == preds).sum()) / len(labels)

class IDTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
        summary_writer: SummaryWriter,
        device: torch.device,
        checkpoint_path: Path,
        checkpoint_frequency: int,
    ):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.summary_writer = summary_writer
        self.step = 0
        self.checkpoint_path = checkpoint_path
        self.checkpoint_frequency = checkpoint_frequency

    def train(
        self,
        epochs: int,
        val_frequency: int,
        print_frequency: int = 20,
        log_frequency: int = 5,
        start_epoch: int = 0
    ):
        self.model.train()
        for epoch in range(start_epoch, epochs):
            self.model.train()
            data_load_start_time = time.time()
            for i, (inputs, targets) in enumerate(self.train_loader):
                batch = inputs.to(self.device)
                labels = targets.to(self.device)
                data_load_end_time = time.time()

                outputs, aux_outputs = self.model(batch)
                loss1 = self.criterion(outputs, labels)
                loss2 = self.criterion(aux_outputs, labels)
                loss = loss1 + 0.4*loss2

                self.optimizer.step()
                self.optimizer.zero_grad()

                with torch.no_grad():
                    preds = outputs.argmax(-1)
                    accuracy = compute_accuracy(labels, preds)

                data_load_time = data_load_end_time - data_load_start_time
                step_time = time.time() - data_load_end_time
                if ((self.step + 1) % log_frequency) == 0:
                    self.log_metrics(epoch, accuracy, loss, data_load_time, step_time)
                if ((self.step + 1) % print_frequency) == 0:
                    self.print_metrics(epoch, accuracy, loss, data_load_time, step_time)

                self.step += 1
                data_load_start_time = time.time()

            if (epoch + 1) % self.checkpoint_frequency == 0 or (epoch + 1) == epochs:
                print(f"Saving model to {self.checkpoint_path}")
                torch.save({"epoch": epoch,
                            "model": self.model.state_dict()},
                            self.checkpoint_path)

            self.summary_writer.add_scalar("epoch", epoch, self.step)
            if ((epoch + 1) % val_frequency) == 0:
                self.validate()
                self.model.train()

    def print_metrics(self, epoch, accuracy, loss, data_load_time, step_time):
        epoch_step = self.step % len(self.train_loader)
        print(
                f"epoch: [{epoch}], "
                f"step: [{epoch_step}/{len(self.train_loader)}], "
                f"batch loss: {loss:.5f}, "
                f"batch accuracy: {accuracy * 100:2.2f}, "
                f"data load time: "
                f"{data_load_time:.5f}, "
                f"step time: {step_time:.5f}"
        )

    def log_metrics(self, epoch, accuracy, loss, data_load_time, step_time):
        self.summary_writer.add_scalar("epoch", epoch, self.step)
        self.summary_writer.add_scalars(
                "accuracy",
                {"train": accuracy},
                self.step
        )
        self.summary_writer.add_scalars(
                "loss",
                {"train": float(loss.item())},
                self.step
        )
        self.summary_writer.add_scalar(
                "time/data", data_load_time, self.step
        )
        self.summary_writer.add_scalar(
                "time/data", step_time, self.step
        )

    def validate(self):
        results = {"preds": [], "labels": []}
        total_loss = 0
        self.model.eval()

        with torch.no_grad():
            for i, (inputs, targets) in enumerate(self.val_loader):
                batch = inputs.to(self.device)
                labels = targets.to(self.device)
                outputs = self.model(batch)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                preds = outputs.argmax(dim=-1).cpu().numpy()
                results["preds"].extend(list(preds))
                results["labels"].extend(list(labels.cpu().numpy()))

        accuracy = compute_accuracy(
            np.array(results["labels"]), np.array(results["preds"])
        )
        average_loss = total_loss / len(self.val_loader)

        print(f"validation loss: {average_loss:.5f}, accuracy: {accuracy * 100:2.2f}")

        self.summary_writer.add_scalars(
                "accuracy",
                {"test": accuracy},
                self.step
        )
        self.summary_writer.add_scalars(
                "loss",
                {"test": average_loss},
                self.step
        )
