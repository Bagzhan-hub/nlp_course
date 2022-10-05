import os
import pickle
import random
from typing import Dict, List, Optional

import numpy as np
import torch


class TextClassificationDataset(Dataset):