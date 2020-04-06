#!/usr/bin/env python3
from multiprocessing import cpu_count

from data.dataset import MantaDataset
from iqa_code.model import CNN
from iqa_code.trainer import Trainer

import torch
import torch.backends.cudnn
from torch import nn, optim
from torch.nn import functional as F
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.datasets
from torchvision import transforms

import argparse
from pathlib import Path
import sys

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(
    description="Train a CNN for Image Quality Assessment on the 100mantas dataset",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--log-dir", default=Path("logs"), type=Path)
parser.add_argument("--learning-rate", default=1e-3, type=float, help="Learning rate")
parser.add_argument("--dropout", default=0, type=float, help="Dropout")
parser.add_argument(
    "--batch-size",
    default=32,
    type=int,
    help="Number of images within each mini-batch",
)
parser.add_argument(
    "--epochs",
    default=50,
    type=int,
    help="Number of epochs (passes through the entire dataset) to train for"
)
parser.add_argument(
    "--model",
    type=str,
    default="resnet-finetuned",
    help="Model to be used for image quality assessment training"
)
parser.add_argument(
    "--val-frequency",
    default=1,
    type=int,
    help="How frequently to test the model on the validation set in number of epochs",
)
parser.add_argument(
    "--log-frequency",
    default=10,
    type=int,
    help="How frequently to save logs to tensorboard in number of steps",
)
parser.add_argument(
    "--print-frequency",
    default=10,
    type=int,
    help="How frequently to print progress to the command line in number of steps",
)
parser.add_argument(
    "-j",
    "--worker-count",
    default=cpu_count(),
    type=int,
    help="Number of worker processes used to load data.",
)
parser.add_argument(
    "--checkpoint-path",
    default=Path("checkpoint.pkl"),
    type=Path,
    help="Provide a file to store checkpoints of the model parameters during training."
)
parser.add_argument(
    "--checkpoint-frequency",
    type=int,
    default=10,
    help="Save a checkpoint every N epochs"
)

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

def main(args):
    train_data_path = 'data/train_data.pkl'
    test_data_path = 'data/test_data.pkl'

    train_loader = torch.utils.data.DataLoader(
        MantaDataset(train_data_path, train=True),
        batch_size=args.batch_size, shuffle=True, num_workers=8,
        pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        MantaDataset(test_data_path, train=False),
        batch_size=args.batch_size, shuffle=True, num_workers=8,
        pin_memory=True
    )

    if args.model == "custom":
        model = CNN(height=512, width=512, channels=3)
    elif args.model in ["resnet-finetuned", "resnet-feature"]:
        model = torchvision.models.resnet18(pretrained=True)
        if args.model == "resnet-feature":
            for param in model.parameters():
                param.requires_grad = False
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 4),
            nn.Dropout(p=args.dropout),
            nn.Sigmoid())

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)

    log_dir = get_summary_writer_log_dir(args)
    print(f"Writing logs to {log_dir}")
    summary_writer = SummaryWriter(
            str(log_dir),
            flush_secs=5
    )

    trainer = Trainer(
        model, train_loader, test_loader, criterion, optimizer,
        summary_writer, DEVICE, args.checkpoint_path, args.checkpoint_frequency
    )

    trainer.train(
        args.epochs,
        args.val_frequency,
        print_frequency=args.print_frequency,
        log_frequency=args.log_frequency,
    )

    summary_writer.close()

def get_summary_writer_log_dir(args: argparse.Namespace) -> str:
    """Get a unique directory that hasn't been logged to before for use with a TB
    SummaryWriter.

    Args:
        args: CLI Arguments

    Returns:
        Subdirectory of log_dir with unique subdirectory name to prevent multiple runs
        from getting logged to the same TB log directory (which you can't easily
        untangle in TB).
    """
    tb_log_dir_prefix = (
        f"IQA_"
        f"model={args.model}_"
        f"bs={args.batch_size}_"
        f"lr={args.learning_rate}_"
        f"dropout={args.dropout}_"
        f"run_"
    i = 0
    while i < 1000:
        tb_log_dir = args.log_dir / (tb_log_dir_prefix + str(i))
        if not tb_log_dir.exists():
            return str(tb_log_dir)
        i += 1
    return str(tb_log_dir)

if __name__ == "__main__":
    main(parser.parse_args())
