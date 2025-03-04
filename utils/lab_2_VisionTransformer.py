import os
import numpy as np
import matplotlib.pyplot as plt
import torch 
from pathlib import Path
from random import sample
from PIL import Image
import math
import sys
from typing import Iterable
from timm.utils import accuracy
import datetime
import time
from collections import defaultdict, deque
import torch.distributed as dist
from torch import nn 
from functools import partial
import torch.nn.functional as F
import timm.models.vision_transformer
from timm.layers import trunc_normal_
from scipy.io import loadmat
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import seaborn as sns
from torch.utils.data import Dataset
import re
import h5py
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import (
    Grayscale, ToTensor, Compose, Resize, InterpolationMode, Normalize, Lambda
)

class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """
    A custom Vision Transformer (ViT) model extending the timm VisionTransformer implementation.

    This class supports:
    - **Global Pooling Strategies**: Allows different pooling methods ('token', 'avg', 'max', etc.).
    - **Tanh Activation**: Applies tanh activation to the output if enabled.
    - **Encoder Freezing**: Allows freezing part or all of the transformer encoder.
    - **Checkpoint Handling**: Supports loading model weights from a checkpoint.

    Attributes:
    ----------
    task : str, required
        Specifies the task ('signal_identification', 'sensing', 'positioning')
    global_pool : str
        The pooling method to apply. Options: 'token', 'avg', 'max', etc.
    tanh : bool
        Applies tanh activation to the final output if set to True.
    """

    def __init__(self, task: str, global_pool: str = "token", tanh: bool = False, **kwargs):
        """
        Initializes the Vision Transformer model.

        Parameters:
        ----------
        global_pool : str, optional
            Specifies the pooling method ('token', 'avg', 'max', etc.), by default "token".
        tanh : bool, optional
            Whether to apply tanh activation to the output, by default False.
        kwargs : dict
            Additional arguments for the base VisionTransformer class.
        """
        super(VisionTransformer, self).__init__(**kwargs)
        self.task = task
        self.global_pool = global_pool
        self.tanh = tanh

    def freeze_encoder(self, num_blocks: int = None):
        """
        Freezes the transformer encoder to prevent updates during training.

        Parameters:
        ----------
        num_blocks : int, optional
            The number of transformer blocks to freeze. If None, freezes the entire encoder.
        """
        if num_blocks is None:
            for param in self.blocks.parameters():
                param.requires_grad = False
        else:
            for param in self.blocks[:num_blocks].parameters():
                param.requires_grad = False

        # Also freeze the patch embedding layer
        for param in self.patch_embed.proj.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Vision Transformer.

        Parameters:
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, channels, height, width).

        Returns:
        -------
        torch.Tensor
            Output tensor after the forward pass.
        """
        x = self.forward_features(x)  # Extracts features using the transformer backbone
        x = self.forward_head(x)  # Passes features through the classifier head

        if self.tanh:
            return torch.tanh(x)  # Applies tanh activation if enabled
        return x

    def save_model(self, path: str):
        """
        Saves the model's state dictionary to a checkpoint file.
    
        Parameters:
        ----------
        path : str
            Path where the model checkpoint will be saved.
        """
        checkpoint = {
            "model": self.state_dict(),  # Save model weights
        }
        
        torch.save(checkpoint, path)
        print(f"Model successfully saved to {path}")

    def load_model(self, checkpoint_path: str) -> dict:
        """
        Loads model weights from a given checkpoint file.

        Parameters:
        ----------
        checkpoint_path : str
            Path to the checkpoint file.

        Returns:
        -------
        dict
            A message indicating the status of the checkpoint loading.
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')  # Load checkpoint
        checkpoint_model = checkpoint['model']  # Extract model weights
        msg = self.load_state_dict(checkpoint_model, strict=True)  # Load state dictionary
        return msg  # Return loading message

    def load_from_pretrained(self, path: str):
        """
        Loads a model from a pretrained checkpoint while handling task-specific modifications.
    
        Parameters:
        ----------
        path : str
            Path to the pretrained model checkpoint.
    
        Returns:
        -------
        msg : dict
            A message from `load_state_dict()` indicating missing/unexpected keys.
        """
        # Load checkpoint
        checkpoint = torch.load(path, map_location='cpu')
        checkpoint_model = checkpoint.get('model', {})
    
        # Get current model's state dictionary
        state_dict = self.state_dict()
    
        # Define keys to be removed if mismatched
        keys_to_remove = ['head.weight', 'head.bias']
        
        if self.task in ['sensing', 'positioning']:
            keys_to_remove.append('pos_embed')
    
        # Remove incompatible keys
        for key in keys_to_remove:
            if key in checkpoint_model and checkpoint_model[key].shape != state_dict[key].shape:
                print(f"Removing key {key} from pretrained checkpoint")
                del checkpoint_model[key]
    
        # Adjust patch embedding projection layer for specific tasks
        patch_embed_key = 'patch_embed.proj.weight'
        if self.task == 'sensing':
            checkpoint_model[patch_embed_key] = checkpoint_model[patch_embed_key].expand(-1, 3, -1, -1)
        elif self.task == 'positioning':
            checkpoint_model[patch_embed_key] = checkpoint_model[patch_embed_key].expand(-1, 4, -1, -1)
    
        # Load state dictionary with `strict=False` to allow missing/unexpected keys
        msg = self.load_state_dict(checkpoint_model, strict=False)
        # Manually initialize fc layer
        trunc_normal_(self.head.weight, std=2e-5)
    
        return msg