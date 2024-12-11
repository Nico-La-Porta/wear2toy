import numpy as np
import pandas as pd
import optuna.visualization as vis
from HumanActivityRecognition.app_config import RAW_DATA_DIR_TRAIN
from HumanActivityRecognition.app_config import RAW_DATA_DIR_TEST

from HumanActivityRecognition.utils import data_preprocessing

from HumanActivityRecognition.run_config import SLIDING_WINDOW_LENGTH
from HumanActivityRecognition.run_config import NB_SENSOR_CHANNELS
from HumanActivityRecognition.run_config import SLIDING_WINDOW_STEP

from HumanActivityRecognition import sliding_window_on_data

from models.DeepConvLSTM import DeepConvLSTM

from HumanActivityRecognition.utils import data_processing
from HumanActivityRecognition import init_weights
from HumanActivityRecognition import train
from HumanActivityRecognition.utils import data_analysis

import optuna

# Impostazione il seed per la riproducibilità
init_weights.set_seed(42)

# Crea il modello
net = DeepConvLSTM()

    
# Prepara i dati
datasetTracesTrain = data_preprocessing.build_dataset(RAW_DATA_DIR_TRAIN)
datasetTracesTest = data_preprocessing.build_dataset(RAW_DATA_DIR_TEST)
dataset_train_labled = data_preprocessing.add_labels_to_dataset(datasetTracesTrain)
dataset_test_labled = data_preprocessing.add_labels_to_dataset(datasetTracesTest)
X_Train, Y_Train = sliding_window_on_data.apply_sliding_window(dataset_train_labled, SLIDING_WINDOW_LENGTH, SLIDING_WINDOW_STEP, NB_SENSOR_CHANNELS)
X_Test, Y_Test = sliding_window_on_data.apply_sliding_window(dataset_test_labled, SLIDING_WINDOW_LENGTH, SLIDING_WINDOW_STEP, NB_SENSOR_CHANNELS)

# Funzione obiettivo per Optuna
def objective(trial):
    # vari iperparametri

    lr = trial.suggest_float('lr', 1e-4, 1e-1, log=True)
    
    # Esegui l'allenamento
    val_loss = train.train(net, X_Train, Y_Train, X_Test, Y_Test, epochs=5, batch_size=84, lr=lr)
    
    return val_loss

# Creazione studio Optuna ottimizza, nel senso di minimizzare la loss in 100 prove
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

print(study.best_params)
vis.plot_optimization_history(study)


"""# Prepara i dati
datasetTracesTrain = data_preprocessing.build_dataset(RAW_DATA_DIR_TRAIN)
datasetTracesTest = data_preprocessing.build_dataset(RAW_DATA_DIR_TEST)
dataset_train_labled = data_preprocessing.add_labels_to_dataset(datasetTracesTrain)
dataset_test_labled = data_preprocessing.add_labels_to_dataset(datasetTracesTest)
X_Train, Y_Train = sliding_window_on_data.apply_sliding_window(dataset_train_labled, SLIDING_WINDOW_LENGTH, SLIDING_WINDOW_STEP, NB_SENSOR_CHANNELS)
X_Test, Y_Test = sliding_window_on_data.apply_sliding_window(dataset_test_labled, SLIDING_WINDOW_LENGTH, SLIDING_WINDOW_STEP, NB_SENSOR_CHANNELS)

net = DeepConvLSTM()
net.apply(init_weights.init_weights)
val_loss = train.train(net, X_Train, Y_Train, X_Test, Y_Test, epochs=5, batch_size=84, lr=0.01)"""
    