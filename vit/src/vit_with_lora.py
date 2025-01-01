import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LoRALayer(nn.Module):
    """Low-Rank Adaptation layer"""
    def __init__(self, in_features, out_features, rank=4, alpha=16):
        super().__init__()