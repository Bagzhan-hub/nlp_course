import os
from typing import Dict, List, Optional

import torch
from omegaconf import DictConfig
from pytorch_lightning import Trainer

__all__ = ['TextClassificationModel']


class TextClassificationModel(object):

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        """Initialize a TextClassificationModel"""
        self.dataset_cfg = cfg.dataset

    def _setup_dataloader_from_config(self, cfg: DictConfig) -> torch.utils.data.DataLoader:
        input_file = cfg.file_path
        if not os.path.exists(input_file):
            raise FileNotFoundError(f'{input_file} not found!')

