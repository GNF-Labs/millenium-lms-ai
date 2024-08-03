import torch
from torch import nn

class UserItemDecoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)