import os
import pandas as pd
import numpy as np
from app_config import REPORTS_DIR, FIGURES_DIR, MODELS_DIR
import sliding_window_on_data
from torch.utils.data import DataLoader
import glob
 
from models.DeepConvLSTM import DeepConvLSTM, HARDataset
import optuna
from optuna.pruners import MedianPruner
import optuna.visualization as vis
import train
import torch
import torch.nn as nn
import train_with_cm
import matplotlib.pyplot as plt
from collections import defaultdict
 
from utils.log_config import logger
from figures import plot_CM
import normalization
 
# Definisco il path da cui leggere i .csv
path = "C:/codes/HumanActivityRecognition/data/downstream_data"
logger.debug(path)
 
# Nome del file CSV finale
final_csv_path = os.path.join(path, 'df_BA_non_null.csv')
 
# Controllo se il file esiste già
if os.path.exists(final_csv_path):
    logger.debug(f"Il file {final_csv_path} esiste già. Lo sto caricando...")
    df_ball = pd.read_csv(final_csv_path)
else:
    # Trovo tutti i file che corrispondono a "BA" nella cartella path e li stampo a schermo
    files = glob.glob(os.path.join(path, "*_BA*.csv"))
    logger.debug(f"Files: {files}")
   
    kid_ball, kid_ball_no_null = [], []
    df_list_ball = []
 
    for file in files:
        df = pd.read_csv(file)
       
        logger.debug(f"Original shape: {df.shape}")
        kid_ball.append(df['kid_id'].unique())
        df = df[df['action_id'] != 0]
        logger.debug(f"Filtered shape: {df.shape}")
        kid_ball_no_null.append(df['kid_id'].unique())
 
        logger.debug(f"Columns: {df.columns}")
        logger.debug(f"Action counts:\n{df['action'].value_counts()}")
        logger.debug(f"Toy counts:\n{df['toy_id'].value_counts()}")
        logger.debug("="*50)
 
    if len(kid_ball) > 0:
        logger.info(f"Numero di utenti che hanno fatto almeno un azione: {len(kid_ball_no_null)/len(kid_ball)}")
    else:
        logger.info("Nessun utente trovato nei file analizzati (kid_ball è vuoto).")
 
    for file in os.listdir(path):
        if not file.endswith('.csv'):
            continue
   
        if file.split('_')[-1].startswith('BA') and file.endswith('.csv'):
            df_temp = pd.read_csv(os.path.join(path, file))
            df_temp = df_temp[df_temp['action_id'] != 0]
            df_list_ball.append(df_temp)
 
    df_ball = pd.concat(df_list_ball)
    logger.info("Dimensioni del df_ball con tutte le attività non nulle")
    logger.info(df_ball.shape)
    logger.info(df_ball.columns)
    logger.info(df_ball['action'].value_counts())
 
    df_ball.to_csv(os.path.join(path, 'df_BA_non_null.csv'), index=False)
 
# Divido il dataframe in base all'attività (action_id) e salvo i dataframe in un file .csv
for action_id in df_ball['action_id'].unique():
    df_action = df_ball[df_ball["action_id"] == action_id]
    logger.debug(f"Dimensioni del dataframe df_ball_action_{action_id} - {df_action.shape}")
    logger.info(f"Conteggio delle attività per df_ball_action_{action_id}")
    df_action.to_csv(os.path.join(path, f'df_ball_action_{action_id}.csv'), index=False)
    logger.debug(f"Salvato il dataframe df_ball_action_{action_id}.csv")
 
# Applico sliding window PRIMA della normalizzazione per calcolare le statistiche sui dati originali
nb_sensor_channels = 9
sliding_window_length = 100
sliding_window_step = 50
 
X, Y = [], []
kid_action_counts = {}
action_files = [f for f in os.listdir(path) if f.startswith('df_ball_action_') and f.endswith('.csv') and not f.endswith('_normalized.csv')]
logger.info(f"File di azioni trovati: {action_files}")
 
for action_file in action_files:
    file_path = os.path.join(path, action_file)
    action_id = action_file.split('_')[-1].replace('.csv', '')
 
    logger.info(f"Processando file: {action_file}")
 
    X_windows, Y_windows, kid_id_action_dict, kid_ids_for_windows = sliding_window_on_data.process_csv(file_path, nb_sensor_channels, sliding_window_length, sliding_window_step)
 
    X.append(X_windows)
    Y.append(Y_windows)
 
    logger.info(f"Numero totale di finestre per l'azione {action_id}: {len(X_windows)}")
    kid_action_counts[f"ball_action_{action_id}"] = kid_id_action_dict
 
# Concateno tutti i dati in un unico array per X e Y
X = np.concatenate(X, axis=0)
Y = np.concatenate(Y, axis=0)
 
logger.info(f"Dimensioni di X finale: {X.shape}")
logger.info(f"Dimensioni di Y finale: {Y.shape}")
 
# Split dei dati in train/val/test
split_ratio_train = 0.7
split_ratio_val = 0.15
split_ratio_test = 0.15
 
X_train, Y_train = [], []
X_val, Y_val = [], []
X_test, Y_test = [], []
 
unique_actions = np.unique(Y)
logger.info("Azioni uniche: %s", unique_actions)
 
train_indices_totali = []
val_indices_totali = []
test_indices_totali = []
 
for action in unique_actions:
    action_indices = np.where(Y == action)[0]
    num_windows = len(action_indices)
    logger.info("Azione %s: %d finestre", action, num_windows)
   
    if num_windows <= 2:
        train_indices = action_indices
        val_indices = np.array([], dtype=int)
        test_indices = np.array([], dtype=int)
    elif num_windows == 3:
        train_indices = action_indices[:1]
        val_indices = action_indices[1:2]
        test_indices = action_indices[2:]
    elif num_windows == 4:
        train_indices = action_indices[:2]
        val_indices = action_indices[2:3]
        test_indices = action_indices[3:]
    elif num_windows == 5:
        train_indices = action_indices[:3]
        val_indices = action_indices[3:4]
        test_indices = action_indices[4:]
    elif num_windows == 6:
        train_indices = action_indices[:4]
        val_indices = action_indices[4:5]
        test_indices = action_indices[5:]
    else:
        num_train = int(num_windows * split_ratio_train)
        num_test = int(num_windows * split_ratio_test)
        num_val = num_windows - num_train - num_test
   
        train_indices = action_indices[:num_train]
        val_indices = action_indices[num_train:num_train + num_val]
        test_indices = action_indices[num_train + num_val:]
 
    logger.info("Azione %s - Train: %d, Val: %d, Test: %d",
               action, len(train_indices), len(val_indices), len(test_indices))
 
    X_train.append(X[train_indices])
    Y_train.append(Y[train_indices])
    X_val.append(X[val_indices])
    Y_val.append(Y[val_indices])
    X_test.append(X[test_indices])
    Y_test.append(Y[test_indices])
 
    train_indices_totali.extend(train_indices.tolist())
    val_indices_totali.extend(val_indices.tolist())
    test_indices_totali.extend(test_indices.tolist())
 
# Concateno tutti i dati
X_train = np.concatenate(X_train, axis=0)
Y_train = np.concatenate(Y_train, axis=0)
X_val = np.concatenate(X_val, axis=0)
Y_val = np.concatenate(Y_val, axis=0)
X_test = np.concatenate(X_test, axis=0)
Y_test = np.concatenate(Y_test, axis=0)
 
Y_train = Y_train.flatten()
Y_val = Y_val.flatten()
Y_test = Y_test.flatten()
 
logger.info("Train shape: %s %s", X_train.shape, Y_train.shape)
logger.info("Val shape: %s %s", X_val.shape, Y_val.shape)
logger.info("Test shape: %s %s", X_test.shape, Y_test.shape)
 
# CALCOLO STATISTICHE DI NORMALIZZAZIONE DAL TRAINING SET DI BALL
# Reshape X_train per calcolare le statistiche per ogni feature
X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])  # (n_samples * window_length, n_features)
logger.info(f"X_train_reshaped shape: {X_train_reshaped.shape}")
 
# Calcolo media e std dal training set di ball
ball_mean = np.mean(X_train_reshaped, axis=0)
ball_std = np.std(X_train_reshaped, axis=0)
ball_std[ball_std == 0] = 1e-6  # Evito divisione per zero
 
logger.info(f"Ball training set - Mean shape: {ball_mean.shape}, Std shape: {ball_std.shape}")
logger.info(f"Ball training set - Mean: {ball_mean}")
logger.info(f"Ball training set - Std: {ball_std}")
 
# Salvo le statistiche del ball training set
ball_stats_dir = os.path.join(REPORTS_DIR, 'ball_stats')
os.makedirs(ball_stats_dir, exist_ok=True)
 
ball_mean_df = pd.DataFrame({'mean': ball_mean})
ball_std_df = pd.DataFrame({'std': ball_std})
ball_mean_df.to_csv(os.path.join(ball_stats_dir, 'ball_mean.csv'), index=False)
ball_std_df.to_csv(os.path.join(ball_stats_dir, 'ball_std.csv'), index=False)
 
logger.info("Statistiche del ball training set salvate")
 
# Normalizzo tutti i dataset usando le statistiche del ball training set
def normalize_data(data, mean, std):
    """Normalizza i dati usando media e std fornite"""
    data_reshaped = data.reshape(-1, data.shape[-1])
    normalized_reshaped = (data_reshaped - mean) / std
    return normalized_reshaped.reshape(data.shape)
 
X_train_normalized = normalize_data(X_train, ball_mean, ball_std)
X_val_normalized = normalize_data(X_val, ball_mean, ball_std)
X_test_normalized = normalize_data(X_test, ball_mean, ball_std)
 
logger.info("Normalizzazione completata usando statistiche del ball training set")
 
# Bilanciamento del test set (come nel codice originale)
train_action_counts = defaultdict(list)
test_action_counts = defaultdict(list)
 
for i, label in enumerate(Y_train):
    train_action_counts[label].append(i)
 
for i, label in enumerate(Y_test):
    test_action_counts[label].append(i)
 
X_train_new, Y_train_new = list(X_train_normalized), list(Y_train)
X_test_new, Y_test_new = list(X_test_normalized), list(Y_test)
 
azioni_modificate = 0
 
for action in np.unique(Y):
    train_ids = train_action_counts.get(action, [])
    test_ids = test_action_counts.get(action, [])
 
    if len(test_ids) == 1 and len(train_ids) >= 3:
        idx_to_move = train_ids[-1]
 
        logger.debug(f"Azione {action} - Sposto la finestra con indice {idx_to_move} dal train al test")
 
        test_indices_totali.append(train_indices_totali[train_ids[-1]])
        train_indices_totali.remove(train_indices_totali[train_ids[-1]])
 
        X_test_new.append(X_train_new[idx_to_move])
        Y_test_new.append(Y_train_new[idx_to_move])
 
        del X_train_new[idx_to_move]
        del Y_train_new[idx_to_move]
 
        azioni_modificate += 1
        logger.info(f"Azione {action}: spostata una finestra da train a test per bilanciare meglio")
 
# Converto di nuovo in numpy
X_train_normalized = np.array(X_train_new)
Y_train = np.array(Y_train_new)
X_test_normalized = np.array(X_test_new)
Y_test = np.array(Y_test_new)
 
logger.info("Azioni modificate: %d", azioni_modificate)
 
# Salvo gli indici finali dopo il bilanciamento
np.savez(f"{path}/split_final_indices_ball_full_finetuning.npz",
         train=np.array(train_indices_totali),
         val=np.array(val_indices_totali),
         test=np.array(test_indices_totali))
 
logger.info("Train shape: %s %s", X_train_normalized.shape, Y_train.shape)
logger.info("Val shape: %s %s", X_val_normalized.shape, Y_val.shape)
logger.info("Test shape: %s %s", X_test_normalized.shape, Y_test.shape)
 
# Mapping delle etichette
unique_labels = np.unique(Y_train)
label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
 
logger.info("Mapping delle etichette: %s", label_mapping)
 
Y_train_mapped = np.array([label_mapping[y] for y in Y_train])
Y_test_mapped = np.array([label_mapping[y] for y in Y_test])
Y_val_mapped = np.array([label_mapping[y] for y in Y_val])
 
logger.info("Nuove etichette train: %s", np.unique(Y_train_mapped))
logger.info("Nuove etichette val: %s", np.unique(Y_val_mapped))
logger.info("Nuove etichette test: %s", np.unique(Y_test_mapped))
 
# Creo i dataset
train_dataset = HARDataset(X_train_normalized, Y_train_mapped)
val_dataset = HARDataset(X_val_normalized, Y_val_mapped)
test_dataset = HARDataset(X_test_normalized, Y_test_mapped)
 
logger.info("Lunghezza dataset di training: %s", len(train_dataset))
logger.info("Lunghezza dataset di validazione: %s", len(val_dataset))
logger.info("Lunghezza dataset di test: %s", len(test_dataset))
 
# Definisco i path per salvare i risultati
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_inference_ball_norm_on_ball_full_finetuning.pkl")
BEST_SCORE_PATH = os.path.join(REPORTS_DIR, "best_score_inference_ball_norm_on_ball_full_finetuning.txt")
 
best_hyperparams_file = os.path.join(REPORTS_DIR, 'best_hyperparameters_inference_ball_norm_on_ball_full_finetuning.csv')
 
if os.path.exists(best_hyperparams_file):
    logger.debug(f"Carico i migliori iperparametri da {best_hyperparams_file}")
    best_hyperparameters = pd.read_csv(best_hyperparams_file).iloc[0].to_dict()
    best_lr = best_hyperparameters['lr']
    best_batch_size = int(best_hyperparameters['batch_size'])
 
else:
    def objective(trial):
        # Definisco gli iperparametri da ottimizzare
        lr = trial.suggest_float('lr', 1e-4, 1e-1, log=True)
        batch_size = trial.suggest_categorical('batch_size', [2, 4, 6])
 
        # Creo i DataLoader con il batch_size suggerito
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
 
        # Creo il modello e carico i pesi pre-addestrati
        model = DeepConvLSTM()
       
        # Carico i pesi pre-addestrati
        model.load_state_dict(
            torch.load(r'C:\codes\HumanActivityRecognition\models\best_model_dl_norm_mean_std_and_aug.pkl',
                      map_location=torch.device('cpu')),
            strict=False
        )
 
        # Sostituisco la testa del modello con la nuova dimensione di classi (4)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 4)  # 4 classi
        model.set_n_classes(4)
 
        # FULL FINE-TUNING: NON congelo nessun parametro
        # Tutti i parametri saranno addestrabili
        for param in model.parameters():
            param.requires_grad = True  # Tutti i pesi saranno addestrabili
 
        logger.info("Modello configurato per FULL FINE-TUNING - tutti i parametri sono addestrabili")
 
        # Eseguo l'allenamento
        best_f1_score = train_with_cm.train(model, train_loader, val_loader, epochs=100, batch_size=batch_size, lr=lr)
 
        # Salvo modello se è il migliore finora
        is_better = False
 
        if not os.path.exists(BEST_SCORE_PATH):
            is_better = True
        else:
            with open(BEST_SCORE_PATH, "r") as f:
                best_score_so_far = float(f.read())
            if best_f1_score > best_score_so_far:
                is_better = True
 
        if is_better:
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            with open(BEST_SCORE_PATH, "w") as f:
                f.write(str(best_f1_score))
            logger.info(f"Nuovo miglior modello salvato in: {BEST_MODEL_PATH} con F1 score: {best_f1_score}")
 
        return best_f1_score
 
    # Creazione studio Optuna
    study = optuna.create_study(
        direction='maximize',
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )
    study.optimize(objective, n_trials=100)
 
    logger.info(f"Best hyperparameters: {study.best_params}")
    logger.info(f"Highest F1-score: {study.best_value}")
 
    # Salvo i best hyperparameters
    best_hyperparameters = study.best_params
    best_hyperparameters['best_f1_score'] = study.best_value
    best_hyperparameters_df = pd.DataFrame([best_hyperparameters])
    best_hyperparameters_df.to_csv(best_hyperparams_file, index=False)
 
    best_lr = study.best_params['lr']
    best_batch_size = study.best_params['batch_size']
 
    # Visualizzo la storia dell'ottimizzazione
    file_name = "optimization_history_inference_ball_norm_on_ball_full_finetuning.png"
    fig = vis.plot_optimization_history(study)
    plt.show()
 
    fig.write_image(os.path.join(FIGURES_DIR, file_name))
    logger.debug(f"Grafico salvato in figures/{file_name}")
 
# Stampo CM dell'allenamento sui vari set
plot_CM(
    mdl_class=DeepConvLSTM,
    mdl_weights=BEST_MODEL_PATH,
    X=X_train_normalized,
    Y=Y_train_mapped,
    batch_size=best_batch_size,
    figure_name="cm_train_best_hyp_inference_ball_norm_on_ball_full_finetuning"
)
 
plot_CM(
    mdl_class=DeepConvLSTM,
    mdl_weights=BEST_MODEL_PATH,
    X=X_val_normalized,
    Y=Y_val_mapped,
    batch_size=best_batch_size,
    figure_name="cm_val_best_hyp_inference_ball_norm_on_ball_full_finetuning"
)
 
# Creo modello e carico pesi del miglior modello per la valutazione finale
model = DeepConvLSTM(n_classes=4)
 
model.load_state_dict(
    torch.load(
        BEST_MODEL_PATH,
        map_location=torch.device('cpu')
    ),
    strict=False
)
 
# Per la valutazione finale, tutti i parametri possono rimanere come sono
# (non serve congelare nulla per l'inferenza)
 
for name, param in model.named_parameters():
    print(f"{name} requires_grad={param.requires_grad}")
 
# Creo i DataLoader con i migliori iperparametri
test_loader = DataLoader(test_dataset, batch_size=best_batch_size, drop_last=True, shuffle=False)
 
# Eseguo l'eval sul test set usando i migliori iperparametri
test_loss, test_acc, test_f1 = train_with_cm.evaluate_model(
    model,
    test_loader,
    figure_name="cm_test_best_hyp_inference_ball_norm_on_ball_full_finetuning",
    save_confusion_matrix=True
)
 
logger.info(f"Test Loss: {test_loss}, Test Accuracy: {test_acc}, Test F1: {test_f1}")