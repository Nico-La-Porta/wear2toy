import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

import normalization
import optuna
import optuna.visualization as vis
from optuna.pruners import MedianPruner

from app_config import PROJ_ROOT, RAW_DATA_DIR_TRAIN, RAW_DATA_DIR_TEST, REPORTS_DIR, FIGURES_DIR, MODELS_DIR
sys.path.append(os.path.join(PROJ_ROOT, "HumanActivityRecognition"))

import train
import train_with_cm
import sliding_window_on_data
from utils.log_config import logger
from utils import data_preprocessing, init_weights
from run_config import SLIDING_WINDOW_LENGTH, NB_SENSOR_CHANNELS, SLIDING_WINDOW_STEP
from models.DeepConvLSTM import DeepConvLSTM, HARDataset

# Impostazione il seed per la riproducibilità
init_weights.set_seed(42)
    
#carico global medians e iqr
global_medians = np.load(os.path.join(REPORTS_DIR, 'global_medians.npy'))
global_iqr = np.load(os.path.join(REPORTS_DIR, 'global_iqr.npy'))

