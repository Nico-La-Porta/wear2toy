import os
import sys
import numpy as np
from app_config import PROJ_ROOT, RAW_DATA_DIR_TRAIN, RAW_DATA_DIR_TEST

sys.path.append(os.path.join(PROJ_ROOT, "HumanActivityRecognition"))
import torch
from utils import data_preprocessing
from utils.log_config import logger

from run_config import SLIDING_WINDOW_LENGTH
from run_config import NB_SENSOR_CHANNELS
from run_config import SLIDING_WINDOW_STEP

import sliding_window_on_data
from torch.utils.data import DataLoader, TensorDataset
from models.DeepConvLSTM import DeepConvLSTM, HARDataset, collate_fn,create_weighted_sampler

from utils.plot import plot_learning_curves
from utils import init_weights
import train_toys_with_cm

import optuna
import optuna.visualization as vis
# Impostazione il seed per la riproducibilità
init_weights.set_seed(42)
    
# Prepara i dati
datasetTracesTrain = data_preprocessing.build_dataset(RAW_DATA_DIR_TRAIN)
datasetTracesTest = data_preprocessing.build_dataset(RAW_DATA_DIR_TEST)
dataset_train_labled = data_preprocessing.add_labels_to_dataset(datasetTracesTrain)
dataset_test_labled = data_preprocessing.add_labels_to_dataset(datasetTracesTest)
X_Train, Y_Train = sliding_window_on_data.apply_sliding_window(dataset_train_labled, SLIDING_WINDOW_LENGTH, SLIDING_WINDOW_STEP, NB_SENSOR_CHANNELS)
X_Test, Y_Test = sliding_window_on_data.apply_sliding_window(dataset_test_labled, SLIDING_WINDOW_LENGTH, SLIDING_WINDOW_STEP, NB_SENSOR_CHANNELS)

# Creazione dataset
train_dataset = HARDataset(X_Train, Y_Train)
test_dataset = HARDataset(X_Test, Y_Test)

# Creazione sampler pesato per il dataset di training
train_sampler = create_weighted_sampler(Y_Train)
# Creazione DataLoader
#train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
#test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, drop_last=True)

train_loader = DataLoader(train_dataset, batch_size=128, drop_last=True,sampler=train_sampler, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, drop_last=True)

# Crea il modello con gli iperparametri suggeriti
net = DeepConvLSTM()

# Esegui l'allenamento
#best_f1_score = train.train(net, train_loader, test_loader, epochs=100, batch_size=128, lr=0.018681199321450608)
best_f1_score= train_toys_with_cm.train(net, train_loader, test_loader, epochs=100, batch_size=128, lr=0.018681199321450608)
print(f"Best F1 score: {best_f1_score}")

torch.save(net.state_dict(), "best_model_dl.pth")

