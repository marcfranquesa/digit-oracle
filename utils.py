import yaml


def get_nns():
    with open("networks.yaml") as f:
        nns = yaml.load(f, Loader=yaml.FullLoader)
    return nns
