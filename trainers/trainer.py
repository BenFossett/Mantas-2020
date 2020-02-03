import time

from utils.accuracies import compute_accuracy

import torch
import torch.backends.cudnn
import numpy as np
from torch import nn
import torchvision.datasets
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from pathlib import Path

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
        summary_writer: SummaryWriter,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.summary_writer = summary_writer
        self.step = 0

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

                logits = self.model.forward(batch)
                labels = labels.float()
                loss = self.criterion(logits, labels)
                loss.backward()

                self.optimizer.step()
                self.optimizer.zero_grad()

                with torch.no_grad():
                    accuracy, _ = compute_accuracy(labels.cpu().numpy(), logits.cpu().numpy())

                data_load_time = data_load_end_time - data_load_start_time
                step_time = time.time() - data_load_end_time
                if ((self.step + 1) % log_frequency) == 0:
                    self.log_metrics(epoch, accuracy, loss, data_load_time, step_time)
                if ((self.step + 1) % print_frequency) == 0:
                    self.print_metrics(epoch, accuracy, loss, data_load_time, step_time)

                self.step += 1
                data_load_start_time = time.time()

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
        results = {"labels": [], "logits": []}
        total_loss = 0
        self.model.eval()

        with torch.no_grad():
            for i, (inputs, targets) in enumerate(self.val_loader):
                batch = inputs.to(self.device)
                labels = targets.to(self.device)
                logits = self.model(batch)
                labels = labels.float()
                loss = self.criterion(logits, labels)
                total_loss += loss.item()
                results["logits"].extend(list(logits.cpu().numpy()))
                results["labels"].extend(list(labels.cpu().numpy()))

        accuracy, label_accuracies = compute_accuracy(
            np.array(results["labels"]), np.array(results["logits"])
        )
        average_loss = total_loss / len(self.val_loader)

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
        labels = ["resolution", "lighting", "pattern", "pose"]
        print(f"validation loss: {average_loss:.5f}, accuracy: {accuracy * 100:2.2f}")
        for i in range(0, len(labels)):
            print("accuracy for " + labels[i] + f": {label_accuracies[i] * 100:2.2f}")
