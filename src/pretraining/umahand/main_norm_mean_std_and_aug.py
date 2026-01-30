import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
# import joblib # use joblib.dump() if it's not a PyTorch model
from torch.utils.data import DataLoader

from src.utils import normalization, train_with_cm
import optuna
import optuna.visualization as vis
from optuna.pruners import MedianPruner

from src.app_config import PROJ_ROOT, RAW_DATA_DIR_TRAIN, RAW_DATA_DIR_TEST, REPORTS_DIR, FIGURES_DIR, MODELS_DIR
sys.path.append(os.path.join(PROJ_ROOT, "HumanActivityRecognition"))



import src.utils.sliding_window_on_data as sliding_window_on_data
from src.utils.log_config import logger
from src.utils import data_preprocessing, init_weights
from src.run_config.run_config import SLIDING_WINDOW_LENGTH, NB_SENSOR_CHANNELS, SLIDING_WINDOW_STEP
from src.models.DeepConvLSTM import DeepConvLSTM, HARDataset
from src.pretraining.figures import plot_CM
from src.utils.transformations import *
from src.utils.transformations_utils import *

# Impostazione il seed per la riproducibilità
init_weights.set_seed(42)

# Definisco i percorsi per i modelli e i migliori risultati
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model_dl_norm_mean_std_and_aug.pkl")
BEST_SCORE_PATH = os.path.join(REPORTS_DIR, "best_score_dl_norm_mean_std_and_aug.txt")
    
# Prepara i dati
datasetTracesTrain = data_preprocessing.build_dataset(RAW_DATA_DIR_TRAIN)
datasetTracesTest = data_preprocessing.build_dataset(RAW_DATA_DIR_TEST)

dataset_train_labled = data_preprocessing.add_labels_to_dataset(datasetTracesTrain)
dataset_test_labled = data_preprocessing.add_labels_to_dataset(datasetTracesTest)


# Concateno i dati di tutte le tracce in un unico array
dataset_train_labled_conc = normalization.concatenate_recordings(dataset_train_labled) #concateno i dati di tutte le tracce in un unico array


mean, std = normalization.compute_mean_std(dataset_train_labled_conc) #calcolo mediana e IQR sul training set

#salvo mean e std per usarli in futuro in formato csv
mean_df = pd.DataFrame(mean, columns=["mean"])
std_df = pd.DataFrame(std, columns=["std"])
mean_df.to_csv(os.path.join(REPORTS_DIR, 'mean_trs.csv'), index=False)
std_df.to_csv(os.path.join(REPORTS_DIR, 'std_trs.csv'), index=False)


scaled_train_data = normalization.normalize_each_recording_mean_std(dataset_train_labled, mean, std) #TODO: sostituisci questa funzione facendo la normalizzazione direttamente nella collate function quando carichi i dati
logger.info("Train data scalers applied")

del dataset_train_labled

scaled_test_data = normalization.normalize_each_recording_mean_std(dataset_test_labled, mean, std)
logger.info("Test data scalers applied")

del dataset_test_labled


X_Train, Y_Train = sliding_window_on_data.apply_sliding_window(scaled_train_data, SLIDING_WINDOW_LENGTH, SLIDING_WINDOW_STEP, NB_SENSOR_CHANNELS)
X_Test, Y_Test = sliding_window_on_data.apply_sliding_window(scaled_test_data, SLIDING_WINDOW_LENGTH, SLIDING_WINDOW_STEP, NB_SENSOR_CHANNELS)

del scaled_train_data, scaled_test_data
print(X_Train.shape)

#DATA AUGUMENTATION
transform_funcs = [
    # transformations.scaling_transform_vectorized, # Use Scaling trasnformation
    noise_transform_vectorized, # Use rotation trasnformation
    scaling_transform_vectorized,
    #rotation_transform_vectorized,
    #axis_angle_to_rotation_matrix_3d_vectorized,
    negate_transform_vectorized,
    time_flip_transform_vectorized,
    intra_sensor_channel_shuffle_transform_vectorized,
    #time_segment_permutation_transform_improved,
    #get_cubic_spline_interpolation,
    time_warp_transform_improved,
    time_warp_transform_low_cost,
]
transformation_function = generate_composite_transform_function_simple(transform_funcs)

tranform_1 = transformation_function(X_Train)
X_Train.shape, tranform_1.shape


print(f"Dimensioni di X_Train: {X_Train.shape}")
print(f"Dimensioni di Y_Train: {Y_Train.shape}")
X_Train_augmented = np.concatenate((X_Train, tranform_1), axis=0)
Y_Train_augmented = np.concatenate((Y_Train, Y_Train), axis=0)
print(f"Dimensioni di X_Train_augmented: {X_Train_augmented.shape}")
print(f"Dimensioni di Y_Train_augmented: {Y_Train_augmented.shape}")

# Creazione dataset
train_dataset = HARDataset(X_Train_augmented, Y_Train_augmented)
test_dataset = HARDataset(X_Test, Y_Test)

del datasetTracesTrain, datasetTracesTest, X_Train, Y_Train, tranform_1

best_hyperparams_file = os.path.join(REPORTS_DIR, 'best_hyperparameters_dl_norm_mean_std_and_aug.csv')
if os.path.exists(best_hyperparams_file):
    logger.debug(f"Carico i migliori iperparametri da {best_hyperparams_file}")
    best_hyperparameters=pd.read_csv(best_hyperparams_file).iloc[0].to_dict()
    best_lr = best_hyperparameters['lr']
    best_batch_size = int(best_hyperparameters['batch_size'])

else:
    logger.debug("Nessun file di iperparametri trovato, eseguo l'ottimizzazione")

    def objective(trial):

        # Suggerisci iperparametri
        lr = trial.suggest_float('lr', 1e-4, 1e-1, log=True)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])

        # Crea DataLoader
        train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

        # Inizializza modello
        net = DeepConvLSTM()

        # Allena il modello
        best_f1_score = train_with_cm.train(net, train_loader, test_loader, epochs=100, lr=lr, exp_figures_dir=FIGURES_DIR, exp_reports_dir=REPORTS_DIR)

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

    # Creazione studio Optuna 
    # Creazione dello studio con il MedianPruner
    study = optuna.create_study(
        direction='maximize', 
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10)  # Parametri di pruning per evitare di continuare trial non promettenti: n_startup_trials=5 significa che i primi 5 trial non verranno prunati, n_warmup_steps=10 significa che dopo 10 trial verrà applicato il pruning
        #Il pruner interromperà automaticamente i trial che non sono promettenti, basandosi sui punteggi parziali (F1-score) ottenuti durante l'allenamento.
    )
    study.optimize(objective, n_trials=100)

    logger.info(f"Highest F1-score: {study.best_value}")
    logger.info(f"Best hyperparameters: {study.best_params}")

    #salvo i best hyperparameters
    best_hyperparameters = study.best_params
    best_hyperparameters['best_f1_score'] = study.best_value
    best_hyperparameters_df = pd.DataFrame([best_hyperparameters])
    best_hyperparameters_df.to_csv(os.path.join(REPORTS_DIR, 'best_hyperparameters_dl_norm_mean_std_and_aug.csv'), index=False)

    best_lr = study.best_params['lr']
    best_batch_size = study.best_params['batch_size']
    #visualizzare la storia dell'ottimizzazione effettuata da Optuna. Ci permette di vedere come l'f1 score
    # è cambiato nel corso delle diverse prove (trials) durante l'ottimizzazione.
    file_name = "optimization_history_dl_norm_mean_std_and_aug.png"
    fig=vis.plot_optimization_history(study)
    plt.show()

    # Salvo il grafico nella cartella FIGURES con il nome specificato
    fig.write_image(os.path.join(FIGURES_DIR, file_name))

    logger.debug(f"Grafico salvato in figures /{file_name}")

#STAMPO CM DELL'ALLENAMENTO SUL TRAINING SET
plot_CM(
    mdl_class=DeepConvLSTM,
    mdl_weights=BEST_MODEL_PATH,
    X=X_Train_augmented,
    Y=Y_Train_augmented,
    batch_size=best_batch_size,
    figure_name="cm_train_best_hyp_optuna_dl_norm_mean_std_and_aug"
)

#STAMPO CM DELL'ALLENAMENTO SUL TESTING SET

plot_CM(
    mdl_class=DeepConvLSTM,
    mdl_weights=BEST_MODEL_PATH,
    X=X_Test,
    Y=Y_Test,
    batch_size=best_batch_size,
    figure_name="cm_test_best_hyp_optuna_dl_norm_mean_std_and_aug"
)
    


# Concateno i datasets
X = np.concatenate((X_Train_augmented, X_Test), axis=0)
Y = np.concatenate((Y_Train_augmented, Y_Test), axis=0)

# Creo un nuovo modello
model = DeepConvLSTM()
# Carico i pesi del miglior modello trovato
model.load_state_dict(torch.load(BEST_MODEL_PATH))
# Imposto il modello in modalità valutazione
model.eval()

del X_Train_augmented, X_Test, Y_Train_augmented, Y_Test, train_dataset, test_dataset

# Creo dataloader
complete_dataset = HARDataset(X, Y)
complete_dataloader = DataLoader(complete_dataset, batch_size=best_batch_size, drop_last=True, shuffle=True)

# Eseguo l'eval sul tutto il dataset usando i migliori iperparametri
test_loss, test_acc, test_f1 = train_with_cm.evaluate_model(model, complete_dataloader, figure_name= "cm_final_eval_best_hyp_optuna_dl_norm_mean_std_and_aug", save_confusion_matrix=True)

logger.info(f"Best model trained with the optimal hyperparameters")