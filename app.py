import streamlit as st
import numpy as np

from utils import get_nn_data, transform
from streamlit_drawable_canvas import st_canvas

import torch
import torch.nn as nn


@st.cache_resource
def load_model(path: str) -> nn.Module:
    return torch.load(path)


def config() -> None:
    st.set_page_config(page_title="Digit Oracle", page_icon="🔮")


def sidebar() -> nn.Module:
    nns = get_nn_data()
    with st.sidebar:
        st.markdown("# About")
        st.markdown(
            """
            App to showcase networks that classify digits. Base code
            developed during the Deep Learning course at GCED, UPC.
            Check my labwork [here](https://github.com/marcfranquesa/gced-coursework/tree/main/AA2).
        """
        )
        st.markdown("---")
        selected = st.selectbox("Network: ", nns.keys())
        network = nns[selected]
        st.markdown(f"**Accuracy**: {network['accuracy']}%")
        st.markdown(f"**Total Parameters**: {network['parameters']}")
    
    return load_model(network["model_path"])


def page(network: nn.Module) -> None:
    st.markdown("# 🔮 Digit Oracle")
    data = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=20,
        stroke_color="black",
        background_color="#ffffff",
        background_image=None,
        update_streamlit=True,
        height=500,
        width=500,
        drawing_mode="freedraw",
        point_display_radius=0,
        display_toolbar=True,
        key="full_app",
    )
    data = transform(data.image_data)
    output = network(data)
    number = output.argmax(dim=1, keepdim=True)[0, 0]

    st.markdown(f"## Predicted: {number}")


def main() -> None:
    config()
    network = sidebar()
    page(network)


if __name__ == "__main__":
    main()
