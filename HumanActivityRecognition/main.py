import os
import sys
import numpy as np
from app_config import PROJ_ROOT, RAW_DATA_DIR_TRAIN, RAW_DATA_DIR_TEST

sys.path.append(os.path.join(PROJ_ROOT, "HumanActivityRecognition"))

from utils import data_preprocessing

from run_config import SLIDING_WINDOW_LENGTH
from run_config import NB_SENSOR_CHANNELS
from run_config import SLIDING_WINDOW_STEP

import sliding_window_on_data
from torch.utils.data import DataLoader
from models.DeepConvLSTM import DeepConvLSTM
from models.DeepConvLSTM import HARDataset
from models.DeepConvLSTM import collate_fn,create_weighted_sampler


from utils import init_weights
import train

from torch.utils.data import DataLoader, TensorDataset

import optuna
from torch.utils.data import DataLoader
import optuna.visualization as vis
# Impostazione il seed per la riproducibilità

import torch
import random

# Imposta i seed per garantire la replicabilità
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)





    
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

#net=DeepConvLSTM()
#train.train(net, train_loader,test_loader,epochs=20, batch_size=32, lr=0.01 )

# Funzione obiettivo per Optuna

def objective(trial):
    # Definisci gli iperparametri da ottimizzare
    lr = trial.suggest_float('lr', 1e-4, 1e-1, log=True)  # Learning rate
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])  # Dimensione del batch
    n_filters = trial.suggest_categorical('n_filters', [32, 64, 128, 256])  # Numero di filtri convoluzionali
    n_hidden = trial.suggest_categorical('n_hidden', [64, 128, 256, 512])  # Numero di unità LSTM
    drop_prob = trial.suggest_float('drop_prob', 0.2, 0.6)  # Probabilità di dropout

    # Crea i DataLoader con il batch_size suggerito
    train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True, sampler=train_sampler, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    # Crea il modello con gli iperparametri suggeriti
    net = DeepConvLSTM(n_filters=n_filters, n_hidden=n_hidden, drop_prob=drop_prob)

    # Esegui l'allenamento e ottieni l'F1 score
    f1_score = train.train(net, train_loader, test_loader, epochs=100, batch_size=batch_size, lr=lr)

    return f1_score

# Creazione studio Optuna per massimizzare l'F1 score in 100 prove
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

print("Best hyperparameters: ", study.best_params)
print("Highest F1 score: ", study.best_value)

# salvare i risultati
study.trials_dataframe().to_csv('results.csv')


#visualizzare la storia dell'ottimizzazione effettuata da Optuna. Ci permette di vedere come la loss
# è cambiata nel corso delle diverse prove (trials) durante l'ottimizzazione.
vis.plot_optimization_history(study)

#salvi i parametri prendo il train e lo ritraini con iperparametri migliori

best_params = study.best_params

# Ricrea i DataLoader con il miglior batch_size
best_batch_size = best_params['batch_size']
best_train_loader = DataLoader(train_dataset, batch_size=best_batch_size, drop_last=True, sampler=train_sampler, collate_fn=collate_fn)
best_test_loader = DataLoader(test_dataset, batch_size=best_batch_size, shuffle=False, drop_last=True)

# Crea il modello con i migliori iperparametri
best_net = DeepConvLSTM(
    n_filters=best_params['n_filters'],
    n_hidden=best_params['n_hidden'],
    drop_prob=best_params['drop_prob']
)

# Addestra il modello finale con i migliori DataLoader
best_f1_score = train.train(
    best_net, best_train_loader, best_test_loader,
    epochs=100, batch_size=best_batch_size, lr=best_params['lr']
)

print(f"Best model F1 score: {best_f1_score:.4f}")



