import random
import sys
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

import networks
from utils import dump_nn_data, get_nn_data


def get_nn_nparams(net: torch.nn.Module) -> int:
    return sum([torch.numel(p) for p in list(net.parameters())])


def get_loaders(
    batch_size: int, test_batch_size: int
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    tfs = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    mnist_trainset = datasets.MNIST(
        root="data", train=True, download=True, transform=tfs
    )
    mnist_testset = datasets.MNIST(
        root="data", train=False, download=True, transform=tfs
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=mnist_trainset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=mnist_testset,
        batch_size=test_batch_size,
        shuffle=False,
        drop_last=True,
    )

    return train_loader, test_loader


def train_epoch(
    train_loader: torch.utils.data.DataLoader,
    network: torch.nn.Module,
    optimizer: torch.optim,
    criterion: torch.nn.functional,
    log_interval: int,
    epoch: int,
) -> Tuple[float, float]:
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print(
                "    Train Epoch: {} [{}/{} ({:.0f}%)]".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                )
            )


def compute_accuracy(predicted_batch: torch.Tensor, label_batch: torch.Tensor) -> int:
    pred = predicted_batch.argmax(dim=1, keepdim=True)
    acum = pred.eq(label_batch.view_as(pred)).sum().item()
    return acum


@torch.no_grad()
def get_accuracy(
    test_loader: torch.utils.data.DataLoader,
    network: torch.nn.Module,
) -> Tuple[float, float]:
    network.eval()
    acc = 0
    for data, target in test_loader:
        output = network(data)
        acc += compute_accuracy(output, target)

    test_acc = 100.0 * acc / len(test_loader.dataset)

    return test_acc


def train(network_data: Dict[str, Any]) -> None:
    network = eval(f"networks.{network_data['network']}()")
    train_loader, test_loader = get_loaders(
        network_data["batch_size"], network_data["test_batch_size"]
    )
    network_data["parameters"] = get_nn_nparams(network)
    criterion = F.nll_loss
    optimizer = torch.optim.RMSprop(
        network.parameters(), lr=network_data["learning_rate"]
    )

    for epoch in range(network_data["num_epochs"]):
        train_epoch(
            train_loader,
            network,
            optimizer,
            criterion,
            network_data["log_interval"],
            epoch,
        )

    network_data["accuracy"] = get_accuracy(test_loader, network)
    torch.save(network, network_data["model_path"])

    return network_data


def main(network: Optional[str] = None) -> None:
    nns = get_nn_data()

    if network == None:
        for name, network_data in nns.items():
            print(f"Starting {name}")
            network_data = train(network_data)
            nns[name] = network_data
        dump_nn_data(nns)
        return

    if network not in nns:
        raise ValueError(f"{network} is not defined")
    network_data = train(nns[network])
    nns[name] = network_data
    dump_nn_data(nns)


if __name__ == "__main__":
    args = sys.argv
    if len(args) == 2:
        main(args[1])
    else:
        main()
