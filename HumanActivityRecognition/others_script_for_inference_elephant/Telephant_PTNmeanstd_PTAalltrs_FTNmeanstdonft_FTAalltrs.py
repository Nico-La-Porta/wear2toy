import os
import pandas as pd
import numpy as np
from app_config import REPORTS_DIR, FIGURES_DIR, MODELS_DIR
import sliding_window_on_data
from torch.utils.data import DataLoader
import glob
from collections import defaultdict
import csv

from models.DeepConvLSTM import DeepConvLSTM, HARDataset 
import optuna
import optuna.visualization as vis
import train
import torch
import train_with_cm
import matplotlib.pyplot as plt
#definisco il path da cui leggere i .csv

from utils.log_config import logger
from utils.data_preprocessing import remove_classes_from_car_data, get_class_distribution

from optuna.pruners import MedianPruner

from figures import plot_CM, create_class_distribution_bar_chart, create_side_by_side_bar_chart
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold

from collections import Counter
from utils import mapping_activity
from utils.transformations import *
from utils.transformations_utils import *
import sys
import normalization


#definisco il path da cui leggere i .csv

path='C:\codes\HumanActivityRecognition\data\downstream_data'
path_1= 'C:\codes\HumanActivityRecognition\data\downstream_data\elep_3022_3024_test'
path_2= 'C:\codes\HumanActivityRecognition\data\downstream_data\elep_3022_3024_test_normalized_pt'
path_3= 'C:\codes\HumanActivityRecognition\data\downstream_data\elep_3022_3024_test_normalized_ft'

#CREO LA CARTELLA SE NON ESISTE
if not os.path.exists(path_1):
    os.makedirs(path_1)
if not os.path.exists(path_2):
    os.makedirs(path_2)

if not os.path.exists(path_3):
    os.makedirs(path_3)
#Visto che l'informazione relativa all'id del bambino lo abbiamo, così come abbiamo anche l'informazione relativa
#al giocattolo, vado a filtrare i .csv in modo tale da avere solo le righe che hanno l'attività non nulla e poi
#appendo tutte le righe delle righe non nulle in un unico dataframe
#per tutti i file che terminano in .csv nella cartella path
# Definisco il percorso della cartella contenente i CSV

# Nome del file CSV finale
final_csv_path = os.path.join(path, 'df_ELEP_non_null.csv')

# Controllo se il file esiste già
if os.path.exists(final_csv_path):
    logger.debug(f"Il file {final_csv_path} esiste già. Lo sto caricando...")
    df_elep = pd.read_csv(final_csv_path)
else:
    suffixes = ["BE", "GE", "RE", "YE", "WE"]
    matching_files=[]
    for suffix in suffixes:
        matching_files.extend(glob.glob(os.path.join(path, f"*_{suffix}*.csv")))
        print(matching_files)

    kid_elep=[] #bambini prima del filtro
    kid_elep_no_null=[] #bambini dopo il filtro
    df_list_elep=[]

    for file in matching_files:
        print("Processing file:", file)
        df = pd.read_csv(file)
        print("Original shape:", df.shape)
        kid_elep.append(df['kid_id'].unique())
        # Filtra le righe con action_id non nullo
        df = df[df['action_id'] != 0]
        print("Filtered shape:", df.shape)
        kid_elep_no_null.append(df['kid_id'].unique())

        print("Columns:", df.columns)
        print("Action counts:\n", df['action'].value_counts())
        print("Toy counts:\n", df['toy_id'].value_counts())
        print("="*50)  # Separatore tra i file

    # mi stampo gli utenti prima di fare il merge e dopo il merge
    logger.info(f"Numero di utenti che hanno fatto almeno un azione: {len(kid_elep_no_null)/len(kid_elep)}")



    for file in matching_files:
        if not file.endswith('.csv'):
            continue
        
        #leggo solo i file che dopo l'undescore ha BE*.csv
        df_temp = pd.read_csv(file)
        df_temp = df_temp[df_temp['action_id'] != 0]  # Mantengo solo righe con attività non nulla
        df_list_elep.append(df_temp)

    # Unisci tutti i dataframe
    df_elep = pd.concat(df_list_elep)

    # Stampa informazioni sul dataframe finale
    print("Dimensioni del dataframe unificato con tutte le attività non nulle")
    print(df_elep.shape)
    print(df_elep.columns)
    print(df_elep['action'].value_counts())

    # Salva il dataframe risultante
    df_elep.to_csv(os.path.join(path, 'df_ELEP_non_null.csv'), index=False)

#filtro le classi per trovare solo quelle che mi interessano
logger.info("FILTRAGGIO CLASSI DAL DATASET ELEPHANT")

# Mostra distribuzione originale
logger.info("Distribuzione action_id originale:")
original_counts = df_elep['action_id'].value_counts().sort_index()
for action_id, count in original_counts.items():
    logger.info(f"   Action {action_id}: {count} righe")


# Rimuovo le classi che non ti interessano
classes_to_remove = [7, 20, 5, 2, 19, 29, 31]
logger.info(f"Rimozione action_id: {classes_to_remove}")

df_elep_filtered = df_elep[~df_elep['action_id'].isin(classes_to_remove)] # filtro il dataframe per rimuovere le classi che non mi interessano

logger.info(f"Righe prima del filtro: {len(df_elep)}")
logger.info(f"Righe dopo il filtro: {len(df_elep_filtered)}")
logger.info(f"Righe rimosse: {len(df_elep) - len(df_elep_filtered)}")


logger.info("Distribuzione action_id dopo filtro:")
filtered_counts = df_elep_filtered['action_id'].value_counts().sort_index()
for action_id, count in filtered_counts.items():
    logger.info(f"   Action {action_id}: {count} righe")


#MI VADO A TENERE DA PARTE IL KID PER IL TEST
KID_TEST_1 = 3022
KID_TEST_2 = 3024


# Divido il dataframe in due: uno per il training/validazione e uno per il test finale (il test finale lo faccio sui due kid) quindi il train non deve contenere Kid_TEST_1 e Kid_TEST_2
df_kid_test = df_elep_filtered[df_elep_filtered['kid_id'].isin([KID_TEST_1, KID_TEST_2])]
df_elep_filtered = df_elep_filtered[~df_elep_filtered['kid_id'].isin([KID_TEST_1, KID_TEST_2])] 





#salvo il df_kid_test
df_kid_test.to_csv(os.path.join(path_1, f'df_test_kid_{KID_TEST_1}_{KID_TEST_2}.csv'), index=False)




#ora divido il dataframe in base all'attività (action_id) e salvo i dataframe in un file .csv
#per ogni attività

for action_id in df_elep_filtered['action_id'].unique():
    df_action = df_elep_filtered[df_elep_filtered["action_id"] == action_id] #filtro il dataframe in base all'attività
    logger.debug(f"Dimensioni del dataframe df_elep_action_{action_id} - {df_action.shape}") #log delle dimensioni del dataframe
    logger.info(f"Conteggio delle attività per df_elep_action_{action_id}") #log del conteggio delle attività
    #salvo il dataframe
    df_action.to_csv(os.path.join(path_1,f'df_elep_action_{action_id}.csv'),index=False) #index=False per non salvare l'indice
    logger.debug(f"Salvato il dataframe df_elep_action_{action_id}.csv")

for action_id in df_kid_test['action_id'].unique():
    df_action = df_kid_test[df_kid_test["action_id"] == action_id] #filtro il dataframe in base all'attività
    logger.debug(f"Dimensioni del dataframe df_kid_test_action_{action_id} - {df_action.shape}") #log delle dimensioni del dataframe
    logger.info(f"Conteggio delle attività per df_kid_test_action_{action_id}") #log del conteggio delle attività
    #salvo il dataframe
    df_action.to_csv(os.path.join(path_1,f'df_kid_test_action_{action_id}.csv'),index=False) #index=False per non salvare l'indice
    logger.debug(f"Salvato il dataframe df_kid_test_action_{action_id}.csv")


#-------------NORMALIZZO DATI RISPETTO AL FINE-TUNING ------------------

means, stds = normalization.compute_dataframe_mean_std(df_elep_filtered)
logger.info(f"Media delle colonne: {means}")
logger.info(f"Deviazione standard delle colonne: {stds}")
df_elep_normalized = normalization.normalize_dataframe_mean_std(df_elep_filtered, means, stds)
df_kid_test_normalized = normalization.normalize_dataframe_mean_std(df_kid_test, means, stds)

#ora divido il dataframe in base all'attività (action_id) e salvo i dataframe in un file .csv
#per ogni attività

for action_id in df_elep_normalized['action_id'].unique():
    df_action = df_elep_normalized[df_elep_normalized["action_id"] == action_id] #filtro il dataframe in base all'attività
    logger.debug(f"Dimensioni del dataframe df_elep_action_{action_id} - {df_action.shape}") #log delle dimensioni del dataframe
    logger.info(f"Conteggio delle attività per df_elep_action_{action_id}") #log del conteggio delle attività
    #salvo il dataframe
    df_action.to_csv(os.path.join(path_3,f'df_elep_action_{action_id}.csv'),index=False) #index=False per non salvare l'indice
    logger.debug(f"Salvato il dataframe df_elep_action_{action_id}.csv")

for action_id in df_kid_test_normalized['action_id'].unique():
    df_action = df_kid_test_normalized[df_kid_test_normalized["action_id"] == action_id] #filtro il dataframe in base all'attività
    logger.debug(f"Dimensioni del dataframe df_kid_test_action_{action_id} - {df_action.shape}") #log delle dimensioni del dataframe
    logger.info(f"Conteggio delle attività per df_kid_test_action_{action_id}") #log del conteggio delle attività
    #salvo il dataframe
    df_action.to_csv(os.path.join(path_3,f'df_kid_test_action_{action_id}.csv'),index=False) #index=False per non salvare l'indice
    logger.debug(f"Salvato il dataframe df_kid_test_action_{action_id}.csv")

logger.info("Normalizzazione completata per tutte le azioni")


#applico sliding window con la funzion process_csv
nb_sensor_channels = 9
sliding_window_length = 100
sliding_window_step = 50

#ora applico la funzione sliding window (che mi da come output x_window e y_window) a tutti i .csv relativi al giocattolo elep
#e poi concateno tutto in un unica x e y 

X, Y = [], []
kid_action_counts = {}


for action_file in [f for f in os.listdir(path_3) if f.endswith('.csv') and f.split('_')[1] == 'elep']:
    file_path = os.path.join(path_3, action_file)
    action = action_file.split('_')[3]

    X_windows, Y_windows, kid_id_action_dict = sliding_window_on_data.process_csv(file_path, nb_sensor_channels, sliding_window_length, sliding_window_step)

    X.append(X_windows)
    Y.append(Y_windows)


    logger.info(f"Numero  totale di finestre per l'azione {action}:{len(X_windows)}")
    kid_action_counts[f"elep_action_{action}"] = kid_id_action_dict #aggiungo il dizionario al dizionario principale per tenere traccia del numero di finestre per ogni bambino per ogni azione, ogni elep_action è una chiave e il valore è un dizionario con il numero di finestre per ogni bambino
logger.info(f"Contenuto finale di kid_action_counts: {kid_action_counts[f'elep_action_{action}']}") #per vedere quante finestre per ogni azione e per ogni bambino sono state elaborate


# Concateno tutti i dati in un unico array per X e Y
X = np.concatenate(X, axis=0)
Y = np.concatenate(Y, axis=0)


# Stampo le dimensioni di X e Y
logger.info(f"Dimensioni di X finale: {X.shape}")
logger.info(f"Dimensioni di Y finale: {Y.shape}")
logger.info(f"Tipo di Y: {Y.dtype}")
logger.info(f"Shape di Y: {Y.shape}")
logger.info(f"Primi 5 elementi di Y: {Y[:5]}")
logger.info(f"Tipo del primo elemento: {type(Y[0])}")


if Y.ndim > 1:
    logger.info(f"Y ha {Y.ndim} dimensioni ({Y.shape}), lo appiattisco...")
    Y = Y.flatten()  # Converte da (2823, 1) a (2823,)
    logger.info(f"Y dopo flatten: {Y.shape}")



logger.info(f"Shape finale di Y: {Y.shape}")
logger.info(f"Primi 5 elementi di Y: {Y[:5]}")
logger.info(f"Tipo del primo elemento: {type(Y[0])}")


logger.info("=== INIZIO ELABORAZIONE TEST KID 3022 E 3024 ===")

X_Test, Y_Test = [], []
kid_action_counts_test = {}

#applico sliding window sul test 
for action_file in [f for f in os.listdir(path_3) if f.endswith('.csv') and f.split('_')[1] == 'kid' and f.split('_')[2] == 'test']:
    file_path = os.path.join(path_3, action_file)
    action = action_file.split('_')[-1].split('.')[0]

    X_windows_test, Y_windows_test, kid_id_action_dict_test = sliding_window_on_data.process_csv(file_path, nb_sensor_channels, sliding_window_length, sliding_window_step)
    X_Test.append(X_windows_test)
    Y_Test.append(Y_windows_test)
    logger.info(f"Numero totale di finestre per l'azione {action} nel test: {len(X_windows_test)}")
    kid_action_counts[f"kid_test_action_{action}"] = kid_id_action_dict
logger.info(f"Contenuto finale di kid_action_counts: {kid_action_counts[f'kid_test_action_{action}']}") #per vedere quante finestre per ogni azione e per ogni bambino sono state elaborate


X_Test = np.concatenate(X_Test, axis=0)
Y_Test = np.concatenate(Y_Test, axis=0)

# Stampo le dimensioni di X_Test e Y_Test
logger.info(f"Dimensioni di X_Test finale: {X_Test.shape}")
logger.info(f"Dimensioni di Y_Test finale: {Y_Test.shape}")
logger.info(f"Tipo di Y_Test: {Y_Test.dtype}")
logger.info(f"Shape di Y_Test: {Y_Test.shape}")


if Y_Test.ndim > 1:
    logger.info(f"Y_Test ha {Y_Test.ndim} dimensioni ({Y_Test.shape}), lo appiattisco...")
    Y_Test = Y_Test.flatten()  # Converte da (2823, 1) a (2823,)
    logger.info(f"Y_Test dopo flatten: {Y_Test.shape}")

logger.info(f"Shape finale di Y_Test: {Y_Test.shape}")
logger.info(f"Primi 5 elementi di Y_Test: {Y_Test[:5]}")


#remap delle etichette per renderle consecutive
unique_labels = np.unique(Y)
label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}

logger.info(f"Mappatura etichette: {label_mapping}")

# Ora il remapping funzionerà correttamente
Y_remapped = np.array([label_mapping[label] for label in Y])

#rempappo anche le etichette del test
Y_Test_remapped = np.array([label_mapping[label] for label in Y_Test])


logger.info("Distribuzione finale delle classi con etichette rimappate:")
get_class_distribution(Y_remapped, "ELEP Dataset con Etichette Rimappate")
get_class_distribution(Y_Test_remapped, "ELEP Dataset Test con Etichette Rimappate")

#uso le Y_remapped per il resto del codice
Y = Y_remapped
Y_Test = Y_Test_remapped


# Usa direttamente il mapping hardcodato
elep_mapping = mapping_activity.ELEPHANT_ACTION_MAPPING
print(f"ELEP mapping keys: {list(elep_mapping.keys())}")

labels_dict = elep_mapping["encoded_to_name"]
print(f"Labels dict: {labels_dict}")

# Ottiengo il numero di classi
num_classes = len(np.unique(Y))
class_names = [elep_mapping["encoded_to_name"][i] for i in range(num_classes)]
print(f"Class names: {class_names}")


# Trovo le classi presenti nel test
test_class_distribution = Counter(Y_Test)
train_class_distribution = Counter(Y)


# Classi assenti nel test
missing_classes = [label for label in np.unique(Y) if label not in test_class_distribution]

logger.info(f"Classi mancanti nel test: {missing_classes}")


# Calcolo quante finestre spostare per ogni classe mancante (15% arrotondato per difetto)
windows_to_move = {}
for missing_class in missing_classes:
    total_windows = train_class_distribution[missing_class]
    windows_to_move[missing_class] = int(round(total_windows * 0.15))  # int() arrotondo per eccesso
    logger.info(f"Classe {missing_class}: {total_windows} finestre totali -> {windows_to_move[missing_class]} finestre da spostare")


# Creo maschere per identificare le finestre da spostare
indices_to_move = []
np.random.seed(42)  # Per riproducibilità


for missing_class in missing_classes:
    # Trovo tutti gli indici delle finestre di questa classe
    class_indices = np.where(Y == missing_class)[0]
    
    # Seleziono casualmente le finestre da spostare
    n_to_move = windows_to_move[missing_class] #mi prendo il numero di finestre da spostare per questa classe
    if n_to_move > 0:
        selected_indices = np.random.choice(class_indices, size=n_to_move, replace=False) #seleziono senza ripetizione n_to_move finestre da spostare
        indices_to_move.extend(selected_indices) #aggiungo gli indici selezionati alla lista degli indici da spostare

# Converto in array numpy
indices_to_move = np.array(indices_to_move)

logger.info(f"Totale finestre da spostare: {len(indices_to_move)}")

# Sposto le finestre dal train al test
if len(indices_to_move) > 0:
    # Estraggo le finestre da spostare
    X_to_move = X[indices_to_move]
    Y_to_move = Y[indices_to_move]
    
    # Rimuovo le finestre dal training set
    mask_keep = np.ones(len(X), dtype=bool)
    mask_keep[indices_to_move] = False
    
    X_train_updated = X[mask_keep]
    Y_train_updated = Y[mask_keep]
    
    # Aggiungo le finestre al test set
    X_Test_updated = np.concatenate([X_Test, X_to_move], axis=0)
    Y_Test_updated = np.concatenate([Y_Test, Y_to_move], axis=0)
    
    # Aggiorno le variabili
    X = X_train_updated
    Y = Y_train_updated
    X_Test = X_Test_updated
    Y_Test = Y_Test_updated
    
    logger.info(f"Finestre spostate con successo!")
    logger.info(f"Nuovo training set: {X.shape}")
    logger.info(f"Nuovo test set: {X_Test.shape}")
    
    # Verifico la nuova distribuzione
    updated_train_distribution = Counter(Y)
    updated_test_distribution = Counter(Y_Test)
    
    logger.info("Nuova distribuzione training:")
    for class_label in sorted(updated_train_distribution.keys()):
        logger.info(f"   Classe {class_label}: {updated_train_distribution[class_label]} finestre")
    
    logger.info("Nuova distribuzione test:")
    for class_label in sorted(updated_test_distribution.keys()):
        logger.info(f"   Classe {class_label}: {updated_test_distribution[class_label]} finestre")
    
    # Verifico che ora tutte le classi siano presenti nel test
    missing_classes_after = [label for label in np.unique(Y) if label not in updated_test_distribution]
    logger.info(f"Classi ancora mancanti nel test: {missing_classes_after}")
    
else:
    logger.info("Nessuna finestra da spostare")

# Aggiorno anche le distribuzioni per il resto del codice
train_class_distribution = Counter(Y)
test_class_distribution = Counter(Y_Test)

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

K_FOLDS = 3


OPTUNA_RESULTS_PATH = os.path.join(REPORTS_DIR, 'kfold_results_Telephant_PTNmeanstd_PTAalltrs_FTNmeanstdonft_FTAalltrs.csv')
OPTUNA_HYPERPARAMS_PATH = os.path.join(REPORTS_DIR, 'best_hyperparameters_kfold_Telephant_PTNmeanstd_PTAalltrs_FTNmeanstdonft_FTAalltrs.csv')
OPTUNA_ALREADY_RUN = os.path.exists(OPTUNA_HYPERPARAMS_PATH)




if OPTUNA_ALREADY_RUN:
    logger.info("=== OPTUNA GIÀ ESEGUITO - CARICO RISULTATI ESISTENTI ===")
    best_hyperparameters_df = pd.read_csv(OPTUNA_HYPERPARAMS_PATH)
    best_hyperparameters = best_hyperparameters_df.iloc[0].to_dict()
    # Carica i risultati dei fold se esistono
    if os.path.exists(OPTUNA_RESULTS_PATH):
        fold_results_df = pd.read_csv(OPTUNA_RESULTS_PATH)
        all_fold_scores = []
        for i, row in fold_results_df.iterrows():
            all_fold_scores.append({
                'fold': row['fold'],
                'params': eval(row['best_params']) if isinstance(row['best_params'], str) else row['best_params'],
                'val_f1': row['best_f1_score']
            })
        best_fold_idx = np.argmax([result['val_f1'] for result in all_fold_scores])
        best_hyperparameters = all_fold_scores[best_fold_idx]['params']
        best_lr_final = best_hyperparameters['lr']
        best_batch_size_final = int(best_hyperparameters['batch_size'])
        fold_f1_scores = fold_results_df['test_f1_score'].tolist()
        mean_f1 = np.mean(fold_f1_scores)
        std_f1 = np.std(fold_f1_scores)
    else:
        logger.warning(f"File dei risultati dei fold non trovato: {OPTUNA_RESULTS_PATH}")
        best_lr_final = best_hyperparameters['lr']
        best_batch_size_final = int(best_hyperparameters['batch_size'])
        mean_f1 = None
        std_f1 = None
        all_fold_scores = [] 
else:
    logger.info("=== INIZIO STRATIFIED K-FOLD CROSS-VALIDATION ===")
    logger.info(f"Numero di fold: {K_FOLDS}")
    # Parametri per K-Fold
    K_FOLDS = 3
    random_state = 42

    # Creo il StratifiedKFold
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=random_state)

    # Dizionario per salvare i risultati di ogni fold
    fold_results = {
        'fold': [],
        'best_f1_score': [],
        'best_params': [],
        'test_f1_score': []
    }

    # Percorsi per i modelli
    BEST_MODEL_KFOLD_DIR = os.path.join(MODELS_DIR, "kfold_elep_models")
    os.makedirs(BEST_MODEL_KFOLD_DIR, exist_ok=True)


    logger.info("=== INIZIO STRATIFIED K-FOLD CROSS-VALIDATION ===")
    logger.info(f"Numero di fold: {K_FOLDS}")


    # Lista per salvare tutti i punteggi per la selezione finale
    all_fold_scores = []


    for fold, (train_val_idx, test_idx) in enumerate(skf.split(X, Y)):
        logger.info(f"\n=== FOLD {fold + 1}/{K_FOLDS} ===")
        
        # Split dei dati per questo fold
        X_trs, Y_trs = X[train_val_idx], Y[train_val_idx]
        X_val, Y_val = X[test_idx], Y[test_idx]

        # Data augmentation SOLO sul train
        X_trs_aug = transformation_function(X_trs)
        Y_trs_aug = Y_trs.copy()

        # Concateno originale e augmentato
        X_trs_final = np.concatenate([X_trs, X_trs_aug], axis=0)
        Y_trs_final = np.concatenate([Y_trs, Y_trs_aug], axis=0)
        
        

        
        logger.info(f"Fold {fold + 1} - Train: {X_trs.shape}, Val: {X_val.shape}")

        # Creo i dataset per questo fold
        train_dataset_fold = HARDataset(X_trs_final, Y_trs_final)
        val_dataset_fold = HARDataset(X_val, Y_val)


        # Ottimizzazione degli iperparametri per questo fold
        def objective_fold(trial):
            lr = trial.suggest_float('lr', 1e-4, 1e-1, log=True)
            batch_size = trial.suggest_categorical('batch_size', [2, 4])
            
            # Creo i DataLoader
            train_loader = DataLoader(train_dataset_fold, batch_size=batch_size, shuffle=True, drop_last=True)
            val_loader = DataLoader(val_dataset_fold, batch_size=batch_size, shuffle=False, drop_last=True)
            
            # Creo il modello
            model = DeepConvLSTM()
            
            # Carico i pesi pre-addestrati
            model.load_state_dict(
                torch.load(r'C:\codes\HumanActivityRecognition\models\best_model_dl_norm_mean_std_and_aug.pkl', 
                        map_location=torch.device('cpu')), 
                strict=False
            )
        
            
            # Sostituisco la testa
            num_ftrs = model.classification_head.in_features
            model.classification_head = nn.Linear(num_ftrs, 16)
            model.set_n_classes(16)

            for param in model.parameters():
                param.requires_grad = True

            
            # Training
            best_f1_score = train_with_cm.train(model, train_loader, val_loader, 
                                            epochs=100, batch_size=batch_size, lr=lr)
            
            return best_f1_score
        
        # Ottimizzazione per questo fold
        logger.info(f"Inizio ottimizzazione iperparametri per fold {fold + 1}")
        study_fold = optuna.create_study(
            direction='maximize',
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        )
        study_fold.optimize(objective_fold, n_trials=50)

        # Salvo i migliori iperparametri per questo fold
        best_params_fold = study_fold.best_params
        best_f1_fold = study_fold.best_value




        # Salvo i risultati
        fold_results['fold'].append(fold + 1)
        fold_results['best_f1_score'].append(best_f1_fold)
        fold_results['best_params'].append(best_params_fold)
        fold_results['test_f1_score'].append(best_f1_fold)
        
        # Aggiungo alla lista per la selezione finale
        all_fold_scores.append({'fold': fold + 1, 'params': best_params_fold, 'val_f1': best_f1_fold})
        
        logger.info(f"Fold {fold + 1} completato - Val F1: {best_f1_fold:.4f}")

# Seleziono i migliori iperparametri
best_fold_idx = np.argmax([result['val_f1'] for result in all_fold_scores])
best_hyperparameters = all_fold_scores[best_fold_idx]['params']
best_lr_final = best_hyperparameters['lr']
best_batch_size_final = int(best_hyperparameters['batch_size'])
#salvo i migliori iperparametri
# Salvo i migliori iperparametri su file CSV
with open(OPTUNA_HYPERPARAMS_PATH, 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=best_hyperparameters.keys())
    writer.writeheader()
    writer.writerow(best_hyperparameters)


with open(OPTUNA_RESULTS_PATH, 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fold_results.keys())
    writer.writeheader()
    # Scrivi una riga per ogni fold
    for i in range(len(fold_results['fold'])):
        row = {key: fold_results[key][i] for key in fold_results}
        writer.writerow(row)
logger.info(f"Risultati dei fold salvati in: {OPTUNA_RESULTS_PATH}")


# Training finale su tutto il train+val (con aug solo sul train)

X_aug = transformation_function(X)
Y_aug = Y.copy()
X_trainval_final = np.concatenate([X, X_aug], axis=0)
Y_trainval_final = np.concatenate([Y, Y_aug], axis=0)
# Creo dataset con tutti i dati
full_dataset = HARDataset(X_trainval_final, Y_trainval_final)

# Creo DataLoader finalE
train_loader_final = DataLoader(full_dataset, batch_size=best_batch_size_final, shuffle=True, drop_last=True)

# Creo modello finale
model_final = DeepConvLSTM()
model_final.load_state_dict(
    torch.load(r'C:\codes\HumanActivityRecognition\models\best_model_dl_norm_mean_std_and_aug.pkl', 
              map_location=torch.device('cpu')), 
    strict=False
)

# Sostituisco la testa
num_ftrs = model_final.classification_head.in_features
model_final.classification_head = nn.Linear(num_ftrs, 16)
model_final.set_n_classes(16)

for param in model_final.parameters():
    param.requires_grad = True


# Training finale
logger.info("Inizio training finale su tutto il train+val...")
final_f1_score = train_with_cm.train(
    model_final, train_loader_final, None,
    epochs=100, batch_size=best_batch_size_final, lr=best_lr_final, validate=False)

# Salvo il modello finale
FINAL_MODEL_PATH = os.path.join(MODELS_DIR, "best_model_kfold_final_Telephant_PTNmeanstd_PTAalltrs_FTNmeanstdonft_FTAalltrs.pkl")
torch.save(model_final.state_dict(), FINAL_MODEL_PATH)

logger.info(f"Modello finale salvato: {FINAL_MODEL_PATH}")

#VALUTAZIONE FINALE ===

# Confusion matrix su tutti i dati
plot_CM(
    mdl_class=DeepConvLSTM,
    mdl_weights=FINAL_MODEL_PATH,
    X=X_trainval_final,
    Y=Y_trainval_final,
    batch_size=best_batch_size_final,
    figure_name="cm_final_kfold_Telephant_PTNmeanstd_PTAalltrs_FTNmeanstdonft_FTAalltrs_all_data_trainval",
    labels_dict=labels_dict,
)

logger.info("=== PROCEDURA K-FOLD COMPLETATA ===")
logger.info(f"Migliori iperparametri: {best_hyperparameters}")
logger.info(f"F1 score medio K-Fold: {mean_f1:.4f} ± {std_f1:.4f}")
logger.info(f"F1 score finale: {final_f1_score:.4f}")



#VALUTO SUL KID MESSO DA PARTE

#STAMPO TIPO DI X TEST E  TEST
logger.info(f"Tipo di X_Test: {type(X_Test)}")
logger.info(f"Dimensioni di X_Test: {X_Test.shape}")
logger.info(f"Tipo di Y_Test: {type(Y_Test)}")
logger.info(f"Dimensioni di Y_Test: {Y_Test.shape}")
test_3009_dataset = HARDataset(X_Test, Y_Test)
test_3009_loader = DataLoader(test_3009_dataset, batch_size=best_batch_size_final, shuffle=False)

# Carica il miglior modello finale e valuta
final_model = DeepConvLSTM()

num_ftrs = model_final.classification_head.in_features
model_final.classification_head = nn.Linear(num_ftrs, 16)
model_final.set_n_classes(16)

final_model.load_state_dict(torch.load(FINAL_MODEL_PATH, map_location='cpu'), strict=False)



# Valutazione
test_loss, test_acc, test_f1 = train_with_cm.evaluate_model(
    final_model, test_3009_loader,
    figure_name=f"cm_holdout_kid_{KID_TEST_1}_{KID_TEST_2}_final_eval_Telephant_PTNmeanstd_PTAalltrs_FTNmeanstdonft_FTAalltrs",
    save_confusion_matrix=True,
    save_f1_score=True,
    labels_dict=labels_dict
)

logger.info(f"Valutazione finale sul kid {KID_TEST_1} and {KID_TEST_2} - F1: {test_f1:.4f}")