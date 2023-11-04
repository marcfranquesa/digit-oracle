import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from streamlit_drawable_canvas import st_canvas

from utils import get_nn_data, transform


@st.cache_resource
def load_model(path: str) -> nn.Module:
    return torch.load(f"src/{path}")


def config() -> None:
    st.set_page_config(page_title="Digit Oracle", page_icon="🔮")


def sidebar() -> nn.Module:
    nns = get_nn_data()
    with st.sidebar:
        st.markdown("# About")
        st.markdown(
            """
            App to showcase networks that classify digits. Networks mainly
            developed during the Deep Learning course at GCED, UPC.
        """
        )
        st.markdown("Made by Marc Franquesa")
        st.markdown("---")
        selected = st.selectbox("Network: ", nns.keys())
        network = nns[selected]
        st.button(
            f"**Accuracy**: {network['accuracy']}%",
            help="Accuracy is measured with the MNIST database",
        )
        st.button(f"**Total Parameters**: {network['parameters']}")

    return load_model(network["model_path"])


def page(network: nn.Module) -> None:
    st.markdown("# 🔮 Digit Oracle")
    data = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=30,
        stroke_color="black",
        background_color="#ffffff",
        background_image=None,
        update_streamlit=True,
        height=350,
        width=350,
        drawing_mode="freedraw",
        point_display_radius=0,
        display_toolbar=True,
        key="full_app",
    )
    if data.image_data is not None:
        data = transform(data.image_data)
        output = network(data)
        number = output.argmax(dim=1, keepdim=True)[0, 0]

        st.markdown(f"## Prediction: {number}")


def main() -> None:
    config()
    network = sidebar()
    page(network)


if __name__ == "__main__":
    main()
