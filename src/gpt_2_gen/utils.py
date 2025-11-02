import torch as tc
import numpy as np
import random
import yaml
from types import SimpleNamespace
from typing import Any

def to_device(
    inputs: dict[str, Any], 
    device: tc.device
) -> dict[str, Any]:
    """
    Moves tensor values in dictionary to specified device.

    Parameters
    ----------
    inputs : dict[str, Any]
        Dictionary of input data containing tensors and possibly
        other types of values.
    device : tc.device
        Device that tensors will be moved to.

    Returns
    -------
    dict[str, Any]
        Returns dictionary with all tensors moved to specified device.
    """
    return {k: (v.to(device) if tc.is_tensor(v) else v) for k, v in inputs.items()}

def get_device(mps: bool=True):
    """
    Finds available devices.

    Checks if Apple Metal Performance Shaders (MPS) is available on the system.
    If True, checks for the MPS device; if not found, defaults to CPU.
    If False, automatically defaults to the CPU.

    Parameters
    ----------
    mps : bool, optional (default=True)
        Allows the function to check for MPS.
        If True, function checks for MPS.
        If False, function defaults to CPU.
        Used for when we always want to use CPU, but 
        do not want to remove the function from the code.
    
    Returns
    -------
    tc.device
        Best available device: `"mps"` if available, otherwise `"cpu"`.
    """
    if mps:
        return tc.device("mps") if tc.backends.mps.is_available() else tc.device("cpu")
    else:
        return tc.device("cpu")

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

def load_config(path: str = "config.yaml") -> dict[str, Any]:
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
