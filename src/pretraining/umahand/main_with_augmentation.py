import torch
import os
import sys

from app_config import PROJ_ROOT, RAW_DATA_DIR_TRAIN, RAW_DATA_DIR_TEST, REPORTS_DIR, FIGURES_DIR, MODELS_DIR
import pandas as pd
sys.path.append(os.path.join(PROJ_ROOT, "HumanActivityRecognition"))


from utils import data_preprocessing
from utils.log_config import logger

from src.run_config.run_config import SLIDING_WINDOW_LENGTH
from src.run_config.run_config import NB_SENSOR_CHANNELS
from src.run_config.run_config import SLIDING_WINDOW_STEP

import src.utils.sliding_window_on_data as sliding_window_on_data
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
from src.pretraining.figures import plot_CM




# Impostazione il seed per la riproducibilità
init_weights.set_seed(42)

# File per salvare i migliori iperparametri
best_hyperparams_file = os.path.join(REPORTS_DIR, 'best_hyperparameters_dl_with_aug.csv')


os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
    
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


train_dataset = HARDataset(X_Train_augmented, Y_Train_augmented)
test_dataset = HARDataset(X_Test, Y_Test)

del dataset_train_labled, dataset_test_labled, datasetTracesTrain, datasetTracesTest, tranform_1
# Definisco i percorsi per i modelli e i migliori risultati
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model_dl_with_aug.pkl")
BEST_SCORE_PATH = os.path.join(REPORTS_DIR, "best_score_dl_with_aug.txt")


best_hyperparams_file = os.path.join(REPORTS_DIR, 'best_hyperparameters_dl_with_aug.csv')
if os.path.exists(best_hyperparams_file):
    logger.debug(f"Carico i migliori iperparametri da {best_hyperparams_file}")
    best_hyperparameters=pd.read_csv(best_hyperparams_file).iloc[0].to_dict()
    best_lr = best_hyperparameters['lr']
    best_batch_size = int(best_hyperparameters['batch_size'])

else:
    print("Nessun file di iperparametri trovato, eseguo l'ottimizzazione")

    def objective(trial):
        # Definisco gli iperparametri da ottimizzare
        lr = trial.suggest_float('lr', 1e-4, 1e-1, log=True)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])


        # Creo i DataLoader con il batch_size suggerito
        train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)


        # Creo il modello con gli iperparametri suggeriti
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
    best_hyperparameters_df.to_csv(os.path.join(REPORTS_DIR, 'best_hyperparameters_dl_with_aug.csv'), index=False)

    best_lr = study.best_params['lr']
    best_batch_size = study.best_params['batch_size']
    #visualizzare la storia dell'ottimizzazione effettuata da Optuna. Ci permette di vedere come l'f1 score
    # è cambiato nel corso delle diverse prove (trials) durante l'ottimizzazione.
    file_name = "optimization_history_dl_with_aug.png"
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
    figure_name="cm_train_best_hyp_optuna_dl_with_aug"
)

#STAMPO CM DELL'ALLENAMENTO SUL TESTING SET

plot_CM(
    mdl_class=DeepConvLSTM,
    mdl_weights=BEST_MODEL_PATH,
    X=X_Test,
    Y=Y_Test,
    batch_size=best_batch_size,
    figure_name="cm_test_best_hyp_optuna_dl_with_aug"
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
test_loss, test_acc, test_f1 = train_with_cm.evaluate_model(model, complete_dataloader, figure_name= "cm_final_eval_best_hyp_optuna_dl_with_aug", save_confusion_matrix=True)

logger.info(f"Best model trained with the optimal hyperparameters")