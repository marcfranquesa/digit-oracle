from typing import Any, Dict, Tuple

import yaml


def get_nn_data() -> Dict[str, Any]:
    with open("networks.yaml") as f:
        nns = yaml.load(f, Loader=yaml.FullLoader)
    return nns


def dump_nn_data(nns_data: Dict[str, Any]) -> None:
    with open("networks.yaml", "w") as f:
        yaml.dump(nns_data, f, default_flow_style=False)
