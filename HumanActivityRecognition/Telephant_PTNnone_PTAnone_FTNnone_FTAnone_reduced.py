import os
import pandas as pd
import numpy as np
from app_config import REPORTS_DIR, FIGURES_DIR, MODELS_DIR
import sliding_window_on_data
from torch.utils.data import DataLoader
import glob
from collections import defaultdict
import csv
from utils.focal_loss import FocalLoss, LabelSmoothingCrossEntropy, WeightedCrossEntropyLoss, CombinedLoss
from sklearn.utils.class_weight import compute_class_weight
from utils.focal_loss import FocalLoss
from sklearn.utils.class_weight import compute_class_weight
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
from figures import plot_CM, combine_kfold_confusion_matrices, create_single_distribution_bar_chart
from optuna.pruners import MedianPruner

from figures import plot_CM, create_class_distribution_bar_chart, create_side_by_side_bar_chart
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold

from collections import Counter
from utils import mapping_activity
from utils.transformations import *
from utils.transformations_utils import *
import sys


#definisco il path da cui leggere i .csv

path='C:\codes\HumanActivityRecognition\data\downstream_data'
path_1= 'C:\codes\HumanActivityRecognition\data\downstream_data\elep_reduced_classes'
#CREO LA CARTELLA SE NON ESISTE
if not os.path.exists(path_1):
    os.makedirs(path_1)
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
KID_TEST_1 = 3008
KID_TEST_2 = 3010
KID_TEST_3 = 3018
KID_TEST_4 = 3022

# Divido il dataframe in due: uno per il training/validazione e uno per il test finale (il test finale lo faccio sui due kid) quindi il train non deve contenere Kid_TEST_1 e Kid_TEST_2
df_kid_test = df_elep_filtered[df_elep_filtered['kid_id'].isin([KID_TEST_1, KID_TEST_2, KID_TEST_3, KID_TEST_4])] # prendo i kid che mi interessano per il test
df_elep_filtered = df_elep_filtered[~df_elep_filtered['kid_id'].isin([KID_TEST_1, KID_TEST_2, KID_TEST_3, KID_TEST_4])] # prendo i kid che mi interessano per il train

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







#applico sliding window con la funzion process_csv
nb_sensor_channels = 9
sliding_window_length = 100
sliding_window_step = 50

#ora applico la funzione sliding window (che mi da come output x_window e y_window) a tutti i .csv relativi al giocattolo elep
#e poi concateno tutto in un unica x e y 

X, Y = [], []
kid_action_counts = {}


for action_file in [f for f in os.listdir(path_1) if f.endswith('.csv') and f.split('_')[1] == 'elep']:
    file_path = os.path.join(path_1, action_file)
    action = action_file.split('_')[-1].split('.')[0]

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


logger.info(f"Dimensioni totali: X={X.shape}, Y={Y.shape}")

logger.info("=== INIZIO ELABORAZIONE TEST KID 3022 E 3024 ===")

X_Test, Y_Test = [], []
kid_action_counts_test = {}

#applico sliding window sul test 
for action_file in [f for f in os.listdir(path_1) if f.endswith('.csv') and f.split('_')[1] == 'kid' and f.split('_')[2] == 'test']:
    file_path = os.path.join(path_1, action_file)
    action = action_file.split('_')[-1].split('.')[0]

    X_windows_test, Y_windows_test, kid_id_action_dict_test = sliding_window_on_data.process_csv(file_path, nb_sensor_channels, sliding_window_length, sliding_window_step)
    X_Test.append(X_windows_test)
    Y_Test.append(Y_windows_test)
    logger.info(f"Numero totale di finestre per l'azione {action} nel test: {len(X_windows_test)}")
    kid_action_counts_test[f"kid_test_action_{action}"] = kid_id_action_dict_test
logger.info(f"Contenuto finale di kid_action_counts_test: {kid_action_counts_test[f'kid_test_action_{action}']}") #per vedere quante finestre per ogni azione e per ogni bambino sono state elaborate


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


# =============================================================================
# STEP 2: RIMOZIONE FINESTRE CON TROPPO PADDING
# =============================================================================
logger.info("=== RIMOZIONE FINESTRE CON TROPPO PADDING ===")

def count_padding_in_window(window):
    """
    Conta il numero di zeri (padding) in una finestra.
    Assume che il padding sia rappresentato da valori 0.
    """
    return np.sum(window == 0)


import numpy as np
from collections import defaultdict

def count_padding_in_window(window):
    """
    Conta il numero di time steps (righe) che sono completamente padding.
    Una riga è considerata padding se TUTTI i suoi valori sono 0.
    """
    if window.ndim == 1:
        # Se la finestra è 1D, conta semplicemente gli zeri
        return np.sum(window == 0)
    else:
        # Se la finestra è 2D (time_steps, features), conta le righe con tutti zeri
        # Una riga è padding se tutti i valori della riga sono 0
        padding_rows = np.all(window == 0, axis=1)
        return np.sum(padding_rows)

def filter_windows_by_padding(X, Y, kid_action_counts, max_windows_per_kid_action=50):
    """
    Filtra le finestre limitando a max_windows_per_kid_action finestre per bambino/azione,
    eliminando prima quelle con più padding.
    
    Args:
        X: array delle finestre (samples, features)
        Y: array delle etichette 
        kid_action_counts: dizionario con struttura {action: {kid_id: count}}
        max_windows_per_kid_action: numero massimo di finestre per bambino/azione
    
    Returns:
        X_filtered, Y_filtered: array filtrati
        kid_action_counts_filtered: dizionario aggiornato
    """
    
    logger.info(f"Inizio filtro finestre. Max finestre per bambino/azione: {max_windows_per_kid_action}")
    
    # Creo un dizionario per mappare ogni finestra al suo bambino e azione
    window_metadata = []
    current_idx = 0
    
    # Ricostruisco la mappatura finestra -> (bambino, azione)
    for action_key, kid_dict in kid_action_counts.items():
        action = action_key.replace('elep_action_', '')
        
        for kid_id, window_count in kid_dict.items():
            for i in range(window_count):
                window_metadata.append({
                    'index': current_idx + i,
                    'kid_id': kid_id,
                    'action': action,
                    'action_key': action_key
                })
            current_idx += window_count
    
    # Calcolo il padding per ogni finestra
    logger.info("Calcolo del padding per ogni finestra...")
    for i, metadata in enumerate(window_metadata):
        padding_count = count_padding_in_window(X[metadata['index']])
        metadata['padding_count'] = padding_count
    
    # Raggruppo per bambino e azione
    kid_action_windows = defaultdict(list)
    for metadata in window_metadata:
        key = f"{metadata['kid_id']}_{metadata['action']}"
        kid_action_windows[key].append(metadata)
    
    # Filtro le finestre per ogni gruppo bambino-azione
    indices_to_keep = []
    kid_action_counts_filtered = {}
    
    for group_key, windows in kid_action_windows.items():
        kid_id, action = group_key.split('_', 1)
        action_key = f"elep_action_{action}"
        
        logger.info(f"Elaboro gruppo {kid_id} - azione {action}: {len(windows)} finestre")
        
        if len(windows) <= max_windows_per_kid_action:
            # Se abbiamo meno finestre del limite, le manteniamo tutte
            indices_to_keep.extend([w['index'] for w in windows])
            
            if action_key not in kid_action_counts_filtered:
                kid_action_counts_filtered[action_key] = {}
            kid_action_counts_filtered[action_key][kid_id] = len(windows)
            
            logger.info(f"  -> Mantenute tutte le {len(windows)} finestre")
            
        else:
            # Se abbiamo più finestre del limite, ordiniamo per padding (decrescente)
            # e prendiamo le prime max_windows_per_kid_action con meno padding
            windows_sorted = sorted(windows, key=lambda x: x['padding_count'])
            selected_windows = windows_sorted[:max_windows_per_kid_action]
            
            indices_to_keep.extend([w['index'] for w in selected_windows])
            
            if action_key not in kid_action_counts_filtered:
                kid_action_counts_filtered[action_key] = {}
            kid_action_counts_filtered[action_key][kid_id] = max_windows_per_kid_action
            
            # Log delle statistiche di padding
            all_padding = [w['padding_count'] for w in windows]
            selected_padding = [w['padding_count'] for w in selected_windows]
            
            logger.info(f"  -> Ridotte da {len(windows)} a {max_windows_per_kid_action} finestre")
            logger.info(f"  -> Padding originale - min: {min(all_padding)}, max: {max(all_padding)}, mean: {np.mean(all_padding):.2f}")
            logger.info(f"  -> Padding selezionato - min: {min(selected_padding)}, max: {max(selected_padding)}, mean: {np.mean(selected_padding):.2f}")
    
    # Ordino gli indici da mantenere
    indices_to_keep.sort()
    
    # Filtro X e Y
    X_filtered = X[indices_to_keep]
    Y_filtered = Y[indices_to_keep]
    
    logger.info(f"Filtro completato:")
    logger.info(f"  -> Finestre originali: {len(X)}")
    logger.info(f"  -> Finestre mantenute: {len(X_filtered)}")
    logger.info(f"  -> Finestre rimosse: {len(X) - len(X_filtered)}")
    
    return X_filtered, Y_filtered, kid_action_counts_filtered

# ============================================================================= 
# APPLICAZIONE DEL FILTRO
# =============================================================================

logger.info("=== APPLICAZIONE FILTRO FINESTRE PER BAMBINO/AZIONE ===")

# Applico il filtro
X_filtered, Y_filtered, kid_action_counts_filtered = filter_windows_by_padding(
    X, Y, kid_action_counts, max_windows_per_kid_action=50
)

#APPLICO il filtro anche sui dati di test
X_Test_filtered, Y_Test_filtered, kid_action_counts_test_filtered = filter_windows_by_padding(
    X_Test, Y_Test, kid_action_counts_test, max_windows_per_kid_action=50
)

# Aggiorno le variabili principali
X = X_filtered
Y = Y_filtered
X_Test = X_Test_filtered
Y_Test = Y_Test_filtered
kid_action_counts = kid_action_counts_filtered
kid_action_counts_test = kid_action_counts_test_filtered

# Log finale
logger.info("=== RISULTATI FINALI DOPO IL FILTRO ===")
logger.info(f"Dimensioni finali: X={X.shape}, Y={Y.shape}")

for action_key, kid_dict in kid_action_counts.items():
    action = action_key.replace('elep_action_', '')
    total_windows = sum(kid_dict.values())
    logger.info(f"Azione {action}: {total_windows} finestre totali")
    for kid_id, count in kid_dict.items():
        logger.info(f"  -> Bambino {kid_id}: {count} finestre")

# Log finale per i dati di test
logger.info("=== RISULTATI FINALI DOPO IL FILTRO SUI DATI DI TEST ===")
logger.info(f"Dimensioni finali test: X_Test={X_Test.shape}, Y_Test={Y_Test.shape}")
for action_key, kid_dict in kid_action_counts_test.items():
    action = action_key.replace('kid_test_action_', '')
    total_windows = sum(kid_dict.values())
    logger.info(f"Azione {action}: {total_windows} finestre totali nel test")
    for kid_id, count in kid_dict.items():
        logger.info(f"  -> Bambino {kid_id}: {count} finestre nel test")

#remap delle etichette per renderle consecutive
unique_labels = np.unique(Y)
label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}

logger.info(f"Mappatura etichette: {label_mapping}")

# Ora il remapping funzionerà correttamente
Y_remapped = np.array([label_mapping[label] for label in Y])

Y_Test_remapped = np.array([label_mapping[label] for label in Y_Test])

logger.info("Distribuzione finale delle classi con etichette rimappate:")
get_class_distribution(Y_remapped, "ELEP Dataset con Etichette Rimappate")
get_class_distribution(Y_Test_remapped, "ELEP Dataset Test con Etichette Rimappate")


#uso le Y_remapped per il resto del codice
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


Y = Y.flatten()
Y_names = [labels_dict[y] for y in Y]
# Crea il bar chart della distribuzione completa
create_single_distribution_bar_chart(
    Y=Y_names,
    toy_name="ELEPHANT",
    title_suffix="",
    save_path=os.path.join(FIGURES_DIR, "elephant_complete_distribution"),
    use_class_prefix=False  
)

# Crea il bar chart della distribuzione completa per il test
Y_Test = Y_Test.flatten()
Y_Test_names = [labels_dict[y] for y in Y_Test]
create_single_distribution_bar_chart(
    Y=Y_Test_names,
    toy_name="ELEPHANT",
    title_suffix=" (Test)",
    save_path=os.path.join(FIGURES_DIR, "elephant_test_distribution"),
    use_class_prefix=False  
)




logger.info("=== CALCOLO PESI DELLE CLASSI PER LOSS FUNCTION ===")

# Calcolo i pesi delle classi per le etichette rimappate
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(Y),
    y=Y
)
class_weights_tensor = torch.FloatTensor(class_weights)
logger.info(f"Pesi delle classi: {dict(zip(np.unique(Y), class_weights))}")

# Mostro distribuzione finale
final_distribution = Counter(Y)
logger.info("Distribuzione finale dopo downsampling:")
for class_id in sorted(final_distribution.keys()):
    weight = class_weights[class_id]
    logger.info(f"   Classe {class_id}: {final_distribution[class_id]} finestre (peso: {weight:.3f})")



K_FOLDS = 3


OPTUNA_RESULTS_PATH = os.path.join(REPORTS_DIR, 'kfold_results_Telephant_PTNnone_PTAnone_FTNnone_FTAnone_reduced.csv')
OPTUNA_HYPERPARAMS_PATH = os.path.join(REPORTS_DIR, 'best_hyperparameters_kfold_Telephant_PTNnone_PTAnone_FTNnone_FTAnone_reduced.csv')
OPTUNA_ALREADY_RUN = os.path.exists(OPTUNA_HYPERPARAMS_PATH)




if OPTUNA_ALREADY_RUN:
    logger.info("=== OPTUNA GIÀ ESEGUITO - CARICO RISULTATI ESISTENTI ===")
    best_hyperparameters_df = pd.read_csv(OPTUNA_HYPERPARAMS_PATH)
    best_hyperparameters = best_hyperparameters_df.iloc[0].to_dict()
    # Conversione dei tipi per i parametri numerici
    if 'batch_size' in best_hyperparameters:
        best_hyperparameters['batch_size'] = int(best_hyperparameters['batch_size'])
    if 'gamma' in best_hyperparameters:
        best_hyperparameters['gamma'] = float(best_hyperparameters['gamma'])
    if 'smoothing' in best_hyperparameters:
        best_hyperparameters['smoothing'] = float(best_hyperparameters['smoothing'])
    if 'lr' in best_hyperparameters:
        best_hyperparameters['lr'] = float(best_hyperparameters['lr'])
    
    # Carica i risultati dei fold se esistono
    if os.path.exists(OPTUNA_RESULTS_PATH):
        fold_results_df = pd.read_csv(OPTUNA_RESULTS_PATH)
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
        
        

        
        logger.info(f"Fold {fold + 1} - Train: {X_trs.shape}, Val: {X_val.shape}")

        # Creo i dataset per questo fold
        train_dataset_fold = HARDataset(X_trs, Y_trs)
        val_dataset_fold = HARDataset(X_val, Y_val)


        # Ottimizzazione degli iperparametri per questo fold
        def objective_fold(trial):
            lr = trial.suggest_categorical('lr', [1e-4, 1e-3, 1e-2], log=True)  # suggest_categorical
            batch_size = trial.suggest_categorical('batch_size', [2, 4])
            # Tipo di loss da utilizzare
            loss_type = trial.suggest_categorical('loss_type', ['focal',
                                                                'label_smoothing',
                                                                'weighted_ce',
                                                                'combined'])
            
            # Parametri specifici per ogni loss
            if loss_type == 'focal':
                gamma = trial.suggest_categorical('gamma', [1.0, 2.0, 3.0])
                criterion = FocalLoss(
                    gamma=gamma, 
                    alpha=1.0, 
                    task_type='multi-class', 
                    num_classes=num_classes
                )
            elif loss_type == 'label_smoothing':
                smoothing = trial.suggest_categorical('smoothing', [0.05, 0.1, 1.5, 0.2])
                criterion = LabelSmoothingCrossEntropy(
                    epsilon=smoothing,
                    reduction='mean'
                )
            elif loss_type == 'combined':
                gamma = trial.suggest_categorical('gamma', [1, 2, 3])
                focal_weight = trial.suggest_categorical('focal_weight', [0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
                ce_weight = 1.0 - focal_weight  # Assicura che i pesi sommino a 1
                criterion = CombinedLoss(
                    focal_weight=focal_weight,
                    ce_weight=ce_weight,
                    gamma=gamma,
                    class_weights=class_weights_tensor,
                    num_classes=num_classes
                )
            else:  # weighted_ce
                criterion = WeightedCrossEntropyLoss(class_weights_tensor)
            # Creo i DataLoader
            train_loader = DataLoader(train_dataset_fold, batch_size=batch_size, shuffle=True, drop_last=True)
            val_loader = DataLoader(val_dataset_fold, batch_size=batch_size, shuffle=False, drop_last=True)
            
            # Creo il modello
            model = DeepConvLSTM()
            
            # Carico i pesi pre-addestrati
            model.load_state_dict(
                torch.load(r'C:\codes\HumanActivityRecognition\models\best_model_dl_without_anything.pkl', 
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
                                            epochs=100, batch_size=batch_size, lr=lr, criterion=criterion)
            
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
# Creo dataset con tutti i dati
full_dataset = HARDataset(X, Y)


# Creo DataLoader finalE
train_loader_final = DataLoader(full_dataset, batch_size=best_batch_size_final, shuffle=True, drop_last=True)

# Creo modello finale
model_final = DeepConvLSTM()
model_final.load_state_dict(
    torch.load(r'C:\codes\HumanActivityRecognition\models\best_model_dl_without_anything.pkl', 
              map_location=torch.device('cpu')), 
    strict=False
)

# Sostituisco la testa
num_ftrs = model_final.classification_head.in_features
model_final.classification_head = nn.Linear(num_ftrs, 16)
model_final.set_n_classes(16)

for param in model_final.parameters():
    param.requires_grad = True


# Creo la loss function ottimale basata sui migliori iperparametri
if best_hyperparameters['loss_type'] == 'focal':
    criterion_final = FocalLoss(
        gamma=best_hyperparameters['gamma'], 
        alpha=1.0, 
        task_type='multi-class', 
        num_classes=num_classes
    )
elif best_hyperparameters['loss_type'] == 'label_smoothing':
    criterion_final = LabelSmoothingCrossEntropy(
        epsilon=best_hyperparameters['smoothing'],
        reduction='mean'
    )
elif best_hyperparameters['loss_type'] == 'combined':
    criterion_final = CombinedLoss(
        focal_weight=best_hyperparameters['focal_weight'],
        ce_weight=1.0 - best_hyperparameters['focal_weight'],
        gamma=best_hyperparameters['gamma'],
        class_weights=class_weights_tensor,
        num_classes=num_classes
    )
else:  # weighted_ce
    criterion_final = WeightedCrossEntropyLoss(class_weights_tensor)

logger.info(f"Loss function finale: {best_hyperparameters['loss_type']}")

# Training finale
logger.info("Inizio training finale su tutto il train+val...")
final_f1_score = train_with_cm.train(
    model_final, train_loader_final, None,
    epochs=100, batch_size=best_batch_size_final, lr=best_lr_final,  criterion=criterion_final, validate=False)

# Salvo il modello finale
FINAL_MODEL_PATH = os.path.join(MODELS_DIR, "best_model_kfold_final_Telephant_PTNnone_PTAnone_FTNnone_FTAnone_reduced.pkl")
torch.save(model_final.state_dict(), FINAL_MODEL_PATH)

logger.info(f"Modello finale salvato: {FINAL_MODEL_PATH}")

#VALUTAZIONE FINALE ===

# Confusion matrix su tutti i dati
plot_CM(
    mdl_class=DeepConvLSTM,
    mdl_weights=FINAL_MODEL_PATH,
    X=X,
    Y=Y,
    batch_size=best_batch_size_final,
    figure_name="cm_final_kfold_Telephant_PTNnone_PTAnone_FTNnone_FTAnone_reduced_all_data_trainval",
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
    figure_name=f"cm_holdout_kid_{KID_TEST_1}_{KID_TEST_2}_final_eval_Telephant_PTNnone_PTAnone_FTNnone_FTAnone_reduced",
    save_confusion_matrix=True,
    save_f1_score=True,
    labels_dict=labels_dict
)

logger.info(f"Valutazione finale sul kid {KID_TEST_1} and {KID_TEST_2} - F1: {test_f1:.4f}")