"""
SCRIPT INFERENCE ON BALL DATASET
- Carico il modello pre-addestrato DeepConvLSTM without anything
- Freezo i pesi del modello
- Sostituisco l'head di classificazione con il numero delle nuove classi
- Ottimizzo gli iperparametri con Optuna su train e val set
- Valuto il modello sul test set
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import glob
import optuna
from optuna.pruners import MedianPruner
import optuna.visualization as vis
from collections import defaultdict

# Import project modules
from app_config import REPORTS_DIR, FIGURES_DIR, MODELS_DIR
from models.DeepConvLSTM import DeepConvLSTM, HARDataset
import sliding_window_on_data
import train_with_cm
from utils.log_config import logger
from figures import plot_CM

# Set paths
DATA_PATH = 'C:/codes/HumanActivityRecognition/data/pdd_data'
PRETRAINED_MODEL_PATH = 'C:/codes/HumanActivityRecognition/models/best_model_dl_without_anything.pkl'
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model_inference_without_anything.pkl")
BEST_SCORE_PATH = os.path.join(REPORTS_DIR, "best_score_inference_without_anything.txt")
BEST_HYPERPARAMS_FILE = os.path.join(REPORTS_DIR, 'best_hyperparameters_inference_without_anything.csv')

# Window parameters for sliding window
NB_SENSOR_CHANNELS = 9
SLIDING_WINDOW_LENGTH = 100
SLIDING_WINDOW_STEP = 50

# Split ratios for train/val/test
SPLIT_RATIO_TRAIN = 0.7
SPLIT_RATIO_VAL = 0.15
SPLIT_RATIO_TEST = 0.15

# Number of classes in the target task
NUM_TARGET_CLASSES = 4





def main():
    """
    Main pipeline for transfer learning with DeepConvLSTM
    """
    logger.info("Starting transfer learning pipeline")
    
    # 1. Load or create merged dataset
    df_ball = load_or_create_merged_dataset(DATA_PATH)
    
    # 2. Split dataset by action
    split_dataset_by_action(df_ball, DATA_PATH)
    
    # 3. Apply sliding window to process data
    X, Y, kid_action_counts = apply_sliding_window(DATA_PATH)
    
    # 4. Create stratified train/val/test split
    X_train, Y_train, X_val, Y_val, X_test, Y_test = create_stratified_split(X, Y, DATA_PATH)
    
    # 5. Balance test set
    X_train, Y_train, X_test, Y_test = balance_test_set(X_train, Y_train, X_test, Y_test)
    
    # 6. Create label mapping
    Y_train_mapped, Y_val_mapped, Y_test_mapped = create_label_mapping(Y_train, Y_val, Y_test)
    
    # 7. Create datasets
    train_dataset, val_dataset, test_dataset = create_datasets_and_loaders(
        X_train, Y_train_mapped, X_val, Y_val_mapped, X_test, Y_test_mapped
    )
    
    # 8. Load best hyperparameters if available or optimize
    if os.path.exists(BEST_HYPERPARAMS_FILE):
        logger.info(f"Loading best hyperparameters from {BEST_HYPERPARAMS_FILE}")
        best_hyperparams = pd.read_csv(BEST_HYPERPARAMS_FILE).iloc[0].to_dict()
    else:
        logger.info("Optimizing hyperparameters with Optuna")
        best_hyperparams = optimize_hyperparameters(train_dataset, val_dataset)
    
    # 9. Evaluate final model
    test_loss, test_acc, test_f1 = evaluate_final_model(
        train_dataset, val_dataset, test_dataset, best_hyperparams
    )
    
    logger.info("Transfer learning pipeline completed successfully")
    return test_loss, test_acc, test_f1


if __name__ == "__main__":
    main()

