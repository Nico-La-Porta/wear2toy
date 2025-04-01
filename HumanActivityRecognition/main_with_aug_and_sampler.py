import torch
import os
import sys

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
from models.DeepConvLSTM import DeepConvLSTM, HARDataset


from utils import init_weights
import train
import train_with_cm

import optuna
import optuna.visualization as vis
from optuna.pruners import MedianPruner
import matplotlib.pyplot as plt
from utils.transformations import *
from utils.transformations_utils import *
from utils.log_config import logger
from models.DeepConvLSTM import DeepConvLSTM, HARDataset, collate_fn, create_weighted_sampler




# Impostazione il seed per la riproducibilità
init_weights.set_seed(42)

# File per salvare i migliori iperparametri
best_hyperparams_file = os.path.join(REPORTS_DIR, 'best_hyperparameters_dl_with_aug_and_sampler.csv')
    
# Prepara i dati
datasetTracesTrain = data_preprocessing.build_dataset(RAW_DATA_DIR_TRAIN)
datasetTracesTest = data_preprocessing.build_dataset(RAW_DATA_DIR_TEST)
dataset_train_labled = data_preprocessing.add_labels_to_dataset(datasetTracesTrain)
dataset_test_labled = data_preprocessing.add_labels_to_dataset(datasetTracesTest)
X_Train, Y_Train = sliding_window_on_data.apply_sliding_window(dataset_train_labled, SLIDING_WINDOW_LENGTH, SLIDING_WINDOW_STEP, NB_SENSOR_CHANNELS)
X_Test, Y_Test = sliding_window_on_data.apply_sliding_window(dataset_test_labled, SLIDING_WINDOW_LENGTH, SLIDING_WINDOW_STEP, NB_SENSOR_CHANNELS)


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
    channel_shuffle_transform_vectorized,
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


train_dataset = HARDataset(X_Train_augmented, Y_Train_augmented)
test_dataset = HARDataset(X_Test, Y_Test)


train_sampler = create_weighted_sampler(Y_Train)

# Controllo se esiste il file con i migliori iperparametri
if os.path.exists(best_hyperparams_file):
    print("Caricamento migliori iperparametri da file CSV...")
    best_hyperparameters = pd.read_csv(best_hyperparams_file).iloc[0].to_dict()
    best_lr = best_hyperparameters['lr']
    best_batch_size = int(best_hyperparameters['batch_size'])  # Optuna salva i numeri interi come float
else:
    print("Nessun file di iperparametri trovato. Avvio Optuna per l'ottimizzazione...")

    # Funzione obiettivo per Optuna
    def objective(trial):
        lr = trial.suggest_float('lr', 1e-4, 1e-1, log=True)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True, sampler=train_sampler, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

        net = DeepConvLSTM()
        best_f1_score = train.train(net, train_loader, test_loader, epochs=100, batch_size=batch_size, lr=lr)
        
        return best_f1_score

    # Creazione studio Optuna con MedianPruner
    study = optuna.create_study(
        direction='maximize', 
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )
    study.optimize(objective, n_trials=50)

    # Salvataggio migliori iperparametri
    best_hyperparameters = study.best_params
    best_hyperparameters['best_f1_score'] = study.best_value
    best_hyperparameters_df = pd.DataFrame([best_hyperparameters])
    best_hyperparameters_df.to_csv(best_hyperparams_file, index=False)

    print("Migliori iperparametri trovati e salvati:", best_hyperparameters)
    best_lr = study.best_params['lr']
    best_batch_size = study.best_params['batch_size']

    # Visualizzazione della storia dell'ottimizzazione
    file_name = "optimization_history_dl_with_aug_and_sampler.png"
    fig = vis.plot_optimization_history(study)
    plt.show()
    fig.write_image(os.path.join(FIGURES_DIR, file_name))
    print(f"Grafico salvato in {FIGURES_DIR}/{file_name}")

# Ora crea e allena il modello con i migliori iperparametri
train_loader = DataLoader(train_dataset, batch_size=best_batch_size, drop_last=True, sampler=train_sampler, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=best_batch_size, shuffle=False, drop_last=True)

best_net = DeepConvLSTM()
best_f1_score = train_with_cm.train(best_net, train_loader, test_loader, epochs=100, batch_size=best_batch_size, lr=best_lr, figure_name="model_dl_with_aug_and_sampler")

# Salva il miglior modello
model_save_path = os.path.join(MODELS_DIR, 'best_model_dl_with_aug_and_sampler.pth')
torch.save(best_net.state_dict(), model_save_path)

print(f"Best model trained with the optimal hyperparameters and saved at {model_save_path}")




