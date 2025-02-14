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
from torch.utils.data import DataLoader, TensorDataset
from models.DeepConvLSTM import DeepConvLSTM, HARDataset, collate_fn,create_weighted_sampler


from utils import init_weights
import train

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
np.save('data\X_Train.npy', X_Train)
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
    lr = trial.suggest_float('lr', 1e-4, 1e-1, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])

    # Crea i DataLoader con il batch_size suggerito
    #runno di nuovo il train con 5 secondi di finestra 
    train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True,sampler=train_sampler, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    # Crea il modello con gli iperparametri suggeriti
    net = DeepConvLSTM()

    # Esegui l'allenamento
    best_f1_score = train.train(net, train_loader, test_loader, epochs=100, batch_size=batch_size, lr=lr)
    
    return best_f1_score

# Creazione studio Optuna ottimizza, nel senso di minimizzare la loss in 100 prove
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

print("Best hyperparameters: ", study.best_params)
print("Highest F1-score: ", study.best_value)

#visualizzare la storia dell'ottimizzazione effettuata da Optuna. Ci permette di vedere come la loss
# è cambiata nel corso delle diverse prove (trials) durante l'ottimizzazione.
vis.plot_optimization_history(study)

#salvi i parametri prendo il train e lo ritraini con iperparametri migliori

#calcolo metriche (confronto le y vere con con il predetto )


#pytorch cross validation