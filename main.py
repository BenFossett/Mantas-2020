#!/usr/bin/env python3
from multiprocessing import cpu_count

from data.dataset import MantaDataset
from models.iqa_model import IQANet
from models.id_model import MantaIDNet
from trainers.iqa_trainer import IQATrainer
from trainers.id_trainer import IDTrainer

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
    description="Train a CNN on the 100mantas dataset",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--log-dir", default=Path("logs"), type=Path)
parser.add_argument("--learning-rate", default=1e-3, type=float, help="Learning rate")
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
    "--mode",
    type=str,
    default="iqa",
    help="What the network is being trained for (iqa, id-full, id-qual)"
)
parser.add_argument(
    "--iqa-model",
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
    if args.mode == "iqa":
        train_data_path = 'data/iqa_train_data.pkl'
        test_data_path = 'data/iqa_test_data.pkl'
    elif args.mode == "id-full":
        train_data_path = 'data/id_train_data.pkl'
        test_data_path = 'data/id_test_data.pkl'
    elif args.mode == "id-qual":
        train_data_path = 'data/qual_train_data.pkl'
        test_data_path = 'data/qual_test_data.pkl'
    else:
        print("Please choose a valid mode.")
        sys.exit(0)

    train_loader = torch.utils.data.DataLoader(
        MantaDataset(train_data_path, mode=args.mode, train=True),
        batch_size=args.batch_size, shuffle=True, num_workers=8,
        pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        MantaDataset(test_data_path, mode=args.mode, train=False),
        batch_size=args.batch_size, shuffle=True, num_workers=8,
        pin_memory=True
    )

    if args.mode == "iqa":
        if args.iqa_model == "custom":
            model = IQANet(height=512, width=512, channels=3)
        elif args.iqa_model in ["resnet-finetuned", "resnet-feature"]:
            model = torchvision.models.resnet18(pretrained=True)
            if args.iqa_model == "resnet-feature":
                for param in model.parameters():
                    param.requires_grad = False
            num_ftrs = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Linear(num_ftrs, 4),
                nn.Sigmoid())
        else:
            print("Please use a valid model.")
            sys.exit(0)

        criterion = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)

    elif args.mode in ["id-full", "id-qual"]:
        #model = MantaIDNet(height=299, width=299, channels=3)
        model = torchvision.models.inception_v3(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        num_ftrs = model.AuxLogits.fc.in_features
        model.AuxLogits.fc = nn.Linear(num_ftrs, 100)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs,100)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    else:
        print("Please choose a valid mode.")
        sys.exit(0)

    log_dir = get_summary_writer_log_dir(args)
    print(f"Writing logs to {log_dir}")
    summary_writer = SummaryWriter(
            str(log_dir),
            flush_secs=5
    )

    if args.mode == "iqa":
        trainer = IQATrainer(
            model, train_loader, test_loader, criterion, optimizer,
            summary_writer, DEVICE, args.checkpoint_path, args.checkpoint_frequency
        )
    else:
        trainer = IDTrainer(
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
        f"CNN_bn_"
        f"mode={args.mode}_"
        f"iqa_model={args.iqa_model}_"
        f"bs={args.batch_size}_"
        f"lr={args.learning_rate}_"
        f"run_"
    )
    i = 0
    while i < 1000:
        tb_log_dir = args.log_dir / (tb_log_dir_prefix + str(i))
        if not tb_log_dir.exists():
            return str(tb_log_dir)
        i += 1
    return str(tb_log_dir)

if __name__ == "__main__":
    main(parser.parse_args())
