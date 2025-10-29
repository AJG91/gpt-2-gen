import torch as tc
import numpy as np
import random
import yaml
from types import SimpleNamespace

from typing import Any, Dict

def to_device(inputs, device):
    """ """
    return {k: (v.to(device) if tc.is_tensor(v) else v) for k, v in inputs.items()}

def get_device():
    """ """
    return tc.device("mps") if tc.backends.mps.is_available() else tc.device("cpu")

def set_seed(seed: int = 42) -> None:
    """
    Sets the random seed across libraries to ensure reproducibility.

    Initializes the random number generators for:
    - PyTorch (CPU and CUDA, if available)
    - NumPy
    - Python's built-in `random` module

    Parameters
    ----------
    seed : int, optional (default=42)
        The seed value to set for all random number generators.
    """
    tc.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if tc.cuda.is_available():
        tc.cuda.manual_seed_all(seed)

def load_config(path: str = "config.yaml") -> Dict[str, Any]:
    """
    Reads a YAML file from the given path and returns its contents as a Python dictionary. 

    Parameters
    ----------
    path : str, optional (default="config.yaml")
        Path to the YAML configuration file.

    Returns
    -------
    dict
        A dictionary containing the configuration parameters parsed
        from the YAML file.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    yaml.YAMLError
        If the file cannot be parsed as valid YAML.
    """
    with open(path) as f:
        return yaml.safe_load(f)
    
def generate_namespace(path: str = "config.yaml") -> SimpleNamespace:
    """
    Generates a parameter namespace using SimpleNamespace.

    Parameters
    ----------
    path : str, optional (default="config.yaml")
        Path to the YAML configuration file.

    Returns
    -------
    SimpleNamespace
        An object whose attributes are the keys found in the config.yaml file.
    """
    config = load_config(path=path)
    return SimpleNamespace(**config)