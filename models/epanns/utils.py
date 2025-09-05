import torch

def move_data_to_device(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    elif isinstance(x, dict):
        return {key: move_data_to_device(value, device) for key, value in x.items()}
    elif isinstance(x, list):
        return [move_data_to_device(value, device) for value in x]
    else:
        return x
