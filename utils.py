from attrdict import AttrDict
import torch as th
import yaml


def load_yaml(yaml_path):
    with open(yaml_path) as f:
        config = yaml.safe_load(f)
    return AttrDict(config)


def sph2cart(phi, theta, r):
    """Conversion from spherical to Cartesian coordinates

    Parameters
    ------
    phi, theta, r: Azimuth angle, zenith angle, distance

    Returns
    ------
    x, y, z : Position in Cartesian coordinates
    """
    x = r * th.sin(theta) * th.cos(phi)
    y = r * th.sin(theta) * th.sin(phi)
    z = r * th.cos(theta)
    return th.hstack((x.unsqueeze(1), y.unsqueeze(1), z.unsqueeze(1)))
