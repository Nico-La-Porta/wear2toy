import os
import sys
import numpy as np
import torch

from app_config import PROJ_ROOT, RAW_DATA_DIR_TRAIN, RAW_DATA_DIR_TEST, REPORTS_DIR, FIGURES_DIR, MODELS_DIR
import pandas as pd
sys.path.append(os.path.join(PROJ_ROOT, "HumanActivityRecognition"))

from utils import data_preprocessing
from utils.log_config import logger

from run_config import SLIDING_WINDOW_LENGTH
from run_config import NB_SENSOR_CHANNELS
from run_config import SLIDING_WINDOW_STEP

import sliding_window_on_data
from torch.utils.data import DataLoader
from models.DeepConvLSTM import DeepConvLSTM, HARDataset, collate_fn, create_weighted_sampler

from utils import init_weights

import optuna
import optuna.visualization as vis
import matplotlib.pyplot as plt
from optuna.pruners import MedianPruner

import train_with_cm
from figures import plot_CM

# Impostazione il seed per la riproducibilità
init_weights.set_seed(42)

# File per salvare i migliori iperparametri
best_hyperparams_file = os.path.join(REPORTS_DIR, 'best_hyperparameters_dl_with_sampler.csv')


# Definisco i percorsi per i modelli e i migliori risultati
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model_dl_with_sampler.pkl")
BEST_SCORE_PATH = os.path.join(REPORTS_DIR, "best_score_dl_with_sampler.txt")

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

del dataset_train_labled, dataset_test_labled, datasetTracesTrain, datasetTracesTest

# Creazione sampler pesato per il dataset di training
train_sampler = create_weighted_sampler(Y_Train)


if os.path.exists(best_hyperparams_file):
    logger.debug(f"Carico i migliori iperparametri da {best_hyperparams_file}")
    best_hyperparameters=pd.read_csv(best_hyperparams_file).iloc[0].to_dict()
    best_lr = best_hyperparameters['lr']
    best_batch_size = int(best_hyperparameters['batch_size'])
else:
    print("Nessun file di iperparametri trovato. Avvio Optuna per l'ottimizzazione...")

    # Funzione obiettivo per Optuna
    def objective(trial):
        lr = trial.suggest_float('lr', 1e-4, 1e-1, log=True)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True, sampler=train_sampler, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

        net = DeepConvLSTM()
        # Eseguo l'allenamento
        best_f1_score = train_with_cm.train(net, train_loader, test_loader, epochs=100, batch_size=batch_size, lr=lr)

        # Salva modello se è il migliore finora
        is_better = False

        if not os.path.exists(BEST_SCORE_PATH):
            is_better = True
        else:
            with open(BEST_SCORE_PATH, "r") as f:
                best_score_so_far = float(f.read())
            if best_f1_score > best_score_so_far:
                is_better = True

        if is_better:
            torch.save(net.state_dict(), BEST_MODEL_PATH)  # use joblib.dump() if it's not a PyTorch model
            with open(BEST_SCORE_PATH, "w") as f:
                f.write(str(best_f1_score))
            logger.info(f"Nuovo miglior modello salvato in: {BEST_MODEL_PATH} con F1 score: {best_f1_score}")

        return best_f1_score

    # Creazione dello studio con il MedianPruner
    study = optuna.create_study(
        direction='maximize', 
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10)  # Parametri di pruning per evitare di continuare trial non promettenti: n_startup_trials=5 significa che i primi 5 trial non verranno prunati, n_warmup_steps=10 significa che dopo 10 trial verrà applicato il pruning
        #Il pruner interromperà automaticamente i trial che non sono promettenti, basandosi sui punteggi parziali (F1-score) ottenuti durante l'allenamento.
    )
    study.optimize(objective, n_trials=100)

    logger.info("Best hyperparameters: ", study.best_params)
    logger.info("Highest F1-score: ", study.best_value)

    #salvo i best hyperparameters
    best_hyperparameters = study.best_params
    best_hyperparameters['best_f1_score'] = study.best_value
    best_hyperparameters_df = pd.DataFrame([best_hyperparameters])
    best_hyperparameters_df.to_csv(os.path.join(REPORTS_DIR, 'best_hyperparameters_dl_with_sampler.csv'), index=False)

    best_lr = study.best_params['lr']
    best_batch_size = study.best_params['batch_size']
    #visualizzare la storia dell'ottimizzazione effettuata da Optuna. Ci permette di vedere come l'f1 score
    # è cambiato nel corso delle diverse prove (trials) durante l'ottimizzazione.
    file_name = "optimization_history_dl_with_sampler.png"
    fig=vis.plot_optimization_history(study)
    plt.show()

    # Salvo il grafico nella cartella FIGURES con il nome specificato
    fig.write_image(os.path.join(FIGURES_DIR, file_name))

    logger.debug(f"Grafico salvato in figures /{file_name}")

#STAMPO CM DELL'ALLENAMENTO SUL TRAINING SET
plot_CM(
    mdl_class=DeepConvLSTM,
    mdl_weights=BEST_MODEL_PATH,
    X=X_Train,
    Y=Y_Train,
    batch_size=best_batch_size,
    figure_name="cm_train_best_hyp_optuna_dl_with_sampler"
)

#STAMPO CM DELL'ALLENAMENTO SUL TESTING SET

plot_CM(
    mdl_class=DeepConvLSTM,
    mdl_weights=BEST_MODEL_PATH,
    X=X_Test,
    Y=Y_Test,
    batch_size=best_batch_size,
    figure_name="cm_test_best_hyp_optuna_dl_with_sampler"
)
    


# Concateno i datasets
X = np.concatenate((X_Train, X_Test), axis=0)
Y = np.concatenate((Y_Train, Y_Test), axis=0)

# Creo un nuovo modello
model = DeepConvLSTM()
# Carico i pesi del miglior modello trovato
model.load_state_dict(torch.load(BEST_MODEL_PATH))
# Imposto il modello in modalità valutazione
model.eval()

del X_Train, X_Test, Y_Train, Y_Test, train_dataset, test_dataset

# Creo dataloader
complete_dataset = HARDataset(X, Y)
complete_dataloader = DataLoader(complete_dataset, batch_size=best_batch_size, drop_last=True, shuffle=True)

# Eseguo l'eval sul tutto il dataset usando i migliori iperparametri
test_loss, test_acc, test_f1 = train_with_cm.evaluate_model(model, complete_dataloader, figure_name= "cm_final_eval_best_hyp_optuna_dl_with_sampler", save_confusion_matrix=True)

logger.info(f"Best model trained with the optimal hyperparameters")

