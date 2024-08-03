import torch
import numpy as np
import torch.nn as nn

class NCF(nn.Module):
    def __init__(self):
        super().__init__()