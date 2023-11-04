from typing import Any, Dict

import numpy as np
import streamlit as st
import torch
import yaml
from skimage.color import rgb2gray
from skimage.transform import resize


def get_nn_data() -> Dict[str, Any]:
    with open("networks.yaml") as f:
        nns = yaml.load(f, Loader=yaml.FullLoader)
    return nns


def dump_nn_data(nns_data: Dict[str, Any]) -> None:
    with open("networks.yaml", "w") as f:
        yaml.dump(nns_data, f, default_flow_style=False)


def transform(image: np.array) -> torch.tensor:
    """Transforms numpy array representing a RGB or RGBA image into
    suitable format for the networks trained
    """
    # image = ((255 - image) / 255 - 0.1307) / 0.3081
    image = rgb2gray(resize(image, (28, 28))[:, :, 0:3])
    image = ((1 - image) / 1 - 0.1307) / 0.3081
    # st.image(image, clamp=True)
    return torch.tensor(image, dtype=torch.float).reshape(1, 1, 28, 28)
