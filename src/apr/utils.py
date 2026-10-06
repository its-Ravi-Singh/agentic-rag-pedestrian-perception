import torch


def get_device():
    """Pick the fastest available device: NVIDIA GPU, then Apple GPU, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
