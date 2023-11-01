import random
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import get_nns


def train(network_data: Dict[str, Any]) -> None:
    network = nn.Sequential(
        *[eval(f"nn.{layer[0]}(*{layer[1]})") for layer in network_data["layers"]]
    )


def main() -> None:
    nns = get_nns()
    for network_data in nns:
        train(network_data)


if __name__ == "__main__":
    main()
