import torch
import torchvision

import torch.nn as nn

from tqdm import tqdm
from torch.utils.data import DataLoader, random_split

from encoder import DualEncoder

class AblationEncoder(DualEncoder):
    def __init__(self, pretrained = True, ablation_type = 'vit'):
        super().__init__(pretrained, ablation_type)