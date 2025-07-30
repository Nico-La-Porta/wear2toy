import os
import pandas as pd
import numpy as np
from app_config import REPORTS_DIR, FIGURES_DIR, MODELS_DIR
import sliding_window_on_data
from torch.utils.data import DataLoader
import glob
from collections import defaultdict

from models.DeepConvLSTM import DeepConvLSTM, HARDataset, collate_fn
import optuna
import optuna.visualization as vis
import train
import torch
import train_with_cm

import matplotlib.pyplot as plt
#definisco il path da cui leggere i .csv

from utils.log_config import logger
from utils.data_preprocessing import  get_class_distribution

from optuna.pruners import MedianPruner

from figures import plot_CM, combine_kfold_confusion_matrices
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from utils import mapping_activity
from utils.data_analysis import plot_clean_activity_timeline_compress, plot_sensor_channels_with_colored_line_and_windows, radar_activity_duration


#definisco il path da cui leggere i .csv

path="C:\codes\HumanActivityRecognition\data\downstream_data"
logger.debug(path)

#CREO SOTTOCARTELLA NELLA CARTELLA FIGURES_DIR CHE SI CHIAMA spoon_figures_lp
figures_spoon_path = os.path.join(FIGURES_DIR, 'spoon_figures_LP')
os.makedirs(figures_spoon_path, exist_ok=True)  # Crea la cartella




#Visto che l'informazione relativa all'id del bambino lo abbiamo, così come abbiamo anche l'informazione relativa
#al giocattolo, vado a filtrare i .csv in modo tale da avere solo le righe che hanno l'attività non nulla e poi
#appendo tutte le righe delle righe non nulle in un unico dataframe
#per tutti i file che terminano in .csv nella cartella path
# Definisco il percorso della cartella contenente i CSV

# Nome del file CSV finale
final_csv_path = os.path.join(path, 'df_SPOON_non_null.csv')

# Controllo se il file esiste già
if os.path.exists(final_csv_path):
    logger.debug(f"Il file {final_csv_path} esiste già. Lo sto caricando...")
    df_spoon = pd.read_csv(final_csv_path)
else:
    #trovo tutti i file che corrispondono a "DO" nella cartella path e li stampo a schermo
    files = glob.glob(os.path.join(path, "*_SP*.csv"))
    logger.debug(f"Files: {files}")
    
    kid_spoon, kid_spoon_no_null = [], [] # liste per salvare utenti prima e dopo il merge 
    df_list_spoon = [] # lista vuota per appendere i dataframe con attività non nulla

    for file in files:
        df = pd.read_csv(file)
        logger.debug(f"Original shape: {df.shape}")
        kid_spoon.append(df['kid_id'].unique())
        df = df[df['action_id'] != 0] #filtro le righe con action_id non nullo
        logger.debug(f"Filtered shape: {df.shape}")
        kid_spoon_no_null.append(df['kid_id'].unique())

        logger.debug(f"Columns: {df.columns}")
        logger.debug(f"Action counts:\n{df['action'].value_counts()}")
        logger.debug(f"Toy counts:\n{df['toy_id'].value_counts()}")
        logger.debug("="*50)  # Separatore tra i file
    
    # mi stampo gli utenti prima di fare il merge e dopo il merge
    if len(kid_spoon) > 0:
        logger.info(f"Numero di utenti che hanno fatto almeno un azione: {len(kid_spoon_no_null)/len(kid_spoon)}")
    else:
        logger.info("Nessun utente trovato nei file analizzati (kid_spoon è vuoto).")

    '''
    Visto che l'informazione relativa all'id del bambino lo abbiamo, così come abbiamo anche l'informazione relativa
    al giocattolo, vado a filtrare i .csv in modo tale da avere solo le righe che hanno l'attività non nulla e poi
    appendo tutte le righe non nulle in un unico dataframe per tutti i file che terminano in .csv nella cartella path
    '''

    for file in os.listdir(path):
        if not file.endswith('.csv'):
            continue
    
        #leggo solo i file che dopo l'undescore ha SP*.csv
        if file.split('_')[-1].startswith('SP') and file.endswith('.csv'): #controllo che il file termini con .csv
            df_temp=pd.read_csv(os.path.join(path,file))
            df_temp = df_temp[df_temp['action_id'] != 0]
            df_list_spoon.append(df_temp)

    df_spoon = pd.concat(df_list_spoon)
    logger.info(f"Shape finale del dataframe: {df_spoon.shape}")
    logger.info(f"Colonne del dataframe: {list(df_spoon.columns)}")
    logger.info(f"Conteggio delle azioni:\n{df_spoon['action'].value_counts()}")

    # salvo il dataframe
    df_spoon.to_csv(os.path.join(path,'df_SPOON_non_null.csv'),index=False)


#ora divido il dataframe in base all'attività (action_id) e salvo i dataframe in un file .csv
#per ogni attività

#filtro le classi per trovare solo quelle che mi interessano
logger.info("FILTRAGGIO CLASSI DAL DATASET SPOON")

# Mostra distribuzione originale
logger.info("Distribuzione action_id originale:")
original_counts = df_spoon['action_id'].value_counts().sort_index()
for action_id, count in original_counts.items():
    logger.info(f"   Action {action_id}: {count} righe")


# Rimuovo le classi che non mi interessano
classes_to_remove = [4]
logger.info(f"Rimozione action_id: {classes_to_remove}")

df_spoon_filtered = df_spoon[~df_spoon['action_id'].isin(classes_to_remove)] # filtro il dataframe per rimuovere le classi che non mi interessano

logger.info(f"Righe prima del filtro: {len(df_spoon)}")
logger.info(f"Righe dopo il filtro: {len(df_spoon_filtered)}")
logger.info(f"Righe rimosse: {len(df_spoon) - len(df_spoon_filtered)}")


logger.info("Distribuzione action_id dopo filtro:")
filtered_counts = df_spoon_filtered['action_id'].value_counts().sort_index()
for action_id, count in filtered_counts.items():
    logger.info(f"   Action {action_id}: {count} righe")


#ora divido il dataframe in base all'attività (action_id) e salvo i dataframe in un file .csv
#per ogni attività

for action_id in df_spoon_filtered['action_id'].unique():
    df_action = df_spoon_filtered[df_spoon_filtered["action_id"] == action_id] #filtro il dataframe in base all'attività
    logger.debug(f"Dimensioni del dataframe df_spoon_action_{action_id} - {df_action.shape}") #log delle dimensioni del dataframe
    logger.info(f"Conteggio delle attività per df_spoon_action_{action_id}") #log del conteggio delle attività
    #salvo il dataframe
    df_action.to_csv(os.path.join(path,f'df_spoon_action_{action_id}.csv'),index=False) #index=False per non salvare l'indice
    logger.debug(f"Salvato il dataframe df_spoon_action_{action_id}.csv")


#applico sliding window con la funzion process_csv
nb_sensor_channels = 9
sliding_window_length = 100
sliding_window_step = 50

#ora applico la funzione sliding window (che mi da come output x_window e y_window) a tutti i .csv relativi al giocattolo spoon
#e poi concateno tutto in un unica x e y 

X, Y = [], []
kid_action_counts = {}
all_consecutivity = []
all_window_indices = []
all_window_row_indices = []
global_window_id = 0

for action_file in [f for f in os.listdir(path) if f.endswith('.csv') and f.startswith('df_spoon_action_') and not f.endswith('normalized_pt.csv')]:
    file_path = os.path.join(path, action_file)
    action = action_file.split('_')[-1].split('.')[0]

    X_windows, Y_windows, kid_id_action_dict, is_consecutive, window_row_indices = sliding_window_on_data.process_csv(file_path, nb_sensor_channels, sliding_window_length, sliding_window_step)

    
    for i in range(len(X_windows)):
        all_window_indices.append({
            'global_window_id': global_window_id,
            'action_id': action,
            #'local_window_id': i,
            #'file': action_file
            'row_indices': window_row_indices[i].tolist()
        })
        global_window_id += 1
    
    X.append(X_windows)
    Y.append(Y_windows)
    all_consecutivity.extend(is_consecutive)
    all_window_row_indices.extend(window_row_indices.tolist())  # <--- aggiungi qui

    logger.info(f"Numero totale di finestre per l'azione {action}: {len(X_windows)}")
    logger.info(f"Finestre consecutive per l'azione {action}: {sum(is_consecutive)}")
    logger.info(f"Finestre non consecutive per l'azione {action}: {len(is_consecutive) - sum(is_consecutive)}")
    logger.info(f"Numero  totale di finestre per l'azione {action}:{len(X_windows)}")
    kid_action_counts[f"spoon_action_{action}"] = kid_id_action_dict #aggiungo il dizionario al dizionario principale per tenere traccia del numero di finestre per ogni bambino per ogni azione, ogni spoon_action è una chiave e il valore è un dizionario con il numero di finestre per ogni bambino
    logger.info(f"Contenuto finale di kid_action_counts: {kid_action_counts}") #per veere quante finestre per ogni azione e per ogni bambino sono state elaborte 


# Concateno tutti i dati in un unico array per X e Y
X = np.concatenate(X, axis=0)
Y = np.concatenate(Y, axis=0)
all_consecutivity = np.array(all_consecutivity)
all_window_row_indices = np.array(all_window_row_indices)  






logger.info(f"Dimensioni di X dopo aggiunta indici finestre: {X.shape}")

logger.info(f"Dimensioni di X dopo concatenazione: {X.shape}")

consecutivity_expanded = all_consecutivity.reshape(-1, 1, 1)  # (n_windows, 1, 1)
consecutivity_repeated = np.repeat(consecutivity_expanded, X.shape[1], axis=1)  # (n_windows, window_length, 1)
# Ora puoi aggiungere le informazioni di consecutività come feature aggiuntiva
X_with_consecutivity = np.concatenate([X, consecutivity_repeated], axis=2)  # (n_windows, window_length, n_features+1)


# Stampo le dimensioni di X e Y
logger.info(f"Dimensioni di X finale: {X.shape}")
logger.info(f"Dimensioni di Y finale: {Y.shape}")
logger.info(f"Tipo di Y: {Y.dtype}")
logger.info(f"Shape di Y: {Y.shape}")
logger.info(f"Primi 5 elementi di Y: {Y[:5]}")
logger.info(f"Tipo del primo elemento: {type(Y[0])}")
# Log delle statistiche
logger.info(f"Finestre totali: {len(all_consecutivity)}")
logger.info(f"Finestre consecutive: {sum(all_consecutivity)} ({sum(all_consecutivity)/len(all_consecutivity)*100:.1f}%)")
logger.info(f"Finestre non consecutive: {len(all_consecutivity) - sum(all_consecutivity)} ({(len(all_consecutivity) - sum(all_consecutivity))/len(all_consecutivity)*100:.1f}%)")

if Y.ndim > 1:
    logger.info(f"Y ha {Y.ndim} dimensioni ({Y.shape}), lo appiattisco...")
    Y = Y.flatten()  # Converte da (2823, 1) a (2823,)
    logger.info(f"Y dopo flatten: {Y.shape}")

logger.info(f"Shape finale di Y: {Y.shape}")
logger.info(f"Primi 5 elementi di Y: {Y[:5]}")
logger.info(f"Tipo del primo elemento: {type(Y[0])}")

#remap delle etichette per renderle consecutive
unique_labels = np.unique(Y)
label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}

logger.info(f"Mappatura etichette: {label_mapping}")

# Ora il remapping funzionerà correttamente
Y_remapped = np.array([label_mapping[label] for label in Y])

logger.info("Distribuzione finale delle classi con etichette rimappate:")
get_class_distribution(Y_remapped, "Test Dataset con Etichette Rimappate")

#uso le Y_remapped per il resto del codice
Y = Y_remapped


# Usa direttamente il mapping hardcodato
spoon_mapping = mapping_activity.SPOON_ACTION_MAPPING
print(f"Spoon mapping keys: {list(spoon_mapping.keys())}")
labels_dict = spoon_mapping["encoded_to_name"]
print(f"Labels dict: {labels_dict}")


class_names = [spoon_mapping["encoded_to_name"][i] for i in range(3)]
print(f"Class names: {class_names}")


logger.info(f"\n=== DEBUG DATASET ===")
logger.info(f"Numero totale finestre: {len(X)}")
logger.info(f"Shape di X: {X.shape}")
logger.info(f"Shape di Y: {Y.shape}")
logger.info(f"Distribuzione classi: {np.bincount(Y)}")



#CREO UN MAPPING RIGA -> FINESTRA: Costruisco un dizionario che per ogni riga del dataframe df_spoon annota 
#a quali global_window_id (id della finestra) appartiene

# Mappo (file, row_index) → lista di window_ids
row_to_window_map = defaultdict(list)

for window_info in all_window_indices:
    window_id = window_info['global_window_id']
    action_id = window_info['action_id']
    row_indices = window_info['row_indices']

    file_name = f'df_spoon_action_{action_id}.csv'
    for idx in row_indices:
        row_to_window_map[(file_name, idx)].append(window_id)



#Aggiungo colonna "window_ids" ai CSV originali :Per ogni file CSV, leggo il DataFrame, 
# aggiungo la colonna "window_ids" (vuota o lista di finestre per ogni riga), e salvo.
# Directory di output per i CSV con la colonna "window_ids"
output_dir = os.path.join(path, "df_with_windows")
os.makedirs(output_dir, exist_ok=True)  # Crea la cartella se non esiste
for action_file in [f for f in os.listdir(path) if f.startswith("df_spoon_action_") and f.endswith(".csv") and not f.endswith('normalized_pt.csv')]:
    full_path = os.path.join(path, action_file)
    df = pd.read_csv(full_path)
    
    # Creo lista di liste di window_ids per ogni riga
    window_id_col = []
    for i in range(len(df)):
        window_ids = row_to_window_map.get((action_file, i), [])
        window_id_col.append(window_ids)

    df["window_ids"] = window_id_col
    
    # Salvo nella nuova cartella
    output_path = os.path.join(output_dir, action_file)
    df.to_csv(output_path, index=False)
    logger.info(f"Aggiunta colonna 'window_ids' e salvato file in: {output_path}")


#ORA AGGIUNGO UNA NUOVA COLONNA  che per ogni riga indica se le finestre a cui appartiene 
# (via "window_ids") sono state predette correttamente o meno, usando i file test_predictions__...csv.

import ast
window_correct_map = {}
# Carico tutte le predizioni in un dizionario window_id → correct
prediction_files = [f for f in os.listdir(figures_spoon_path) if f.startswith("test_predictions__") and f.endswith(".csv")]
for prediction_file in prediction_files:
    prediction_df = pd.read_csv(os.path.join(figures_spoon_path, prediction_file))
    
    for _, row in prediction_df.iterrows():
        try:
            window_info = ast.literal_eval(row["window_index"])
            global_id = window_info['global_window_id']
            window_correct_map[global_id] = row["correct"]
        except Exception as e:
            logger.warning(f"Errore nel file {prediction_file} alla riga {row.name}: {e}")

logger.info(f"Totale finestre uniche trovate nelle predizioni: {len(window_correct_map)}")

# Aggiorno CSV in df_with_windows aggiungendo correct info
input_dir = os.path.join(path, "df_with_windows")
output_dir = os.path.join(path, "df_with_windows_with_correct")
os.makedirs(output_dir, exist_ok=True)

for file in os.listdir(input_dir):
    if not file.startswith("df_spoon_action_") or not file.endswith(".csv"):
        continue

    df = pd.read_csv(os.path.join(input_dir, file))

    # Converti stringa "[0, 1]" in lista vera
    df["window_ids"] = df["window_ids"].apply(ast.literal_eval)

    # Aggiungi lista dei correct associati a ogni riga
    df["correct_windows"] = df["window_ids"].apply(
        lambda ids: [window_correct_map.get(wid, None) for wid in ids]
    )

    # Sintesi: tutte corrette / alcune sbagliate / nessuna finestra
    def correct_summary(c_list):
        if not c_list: return "no_windows"
        if all(c is True for c in c_list): return "all_correct"
        if any(c is False for c in c_list): return "some_incorrect"
        return "unknown"

    df["correct_summary"] = df["correct_windows"].apply(correct_summary)

    # (facoltativo) True se almeno una è sbagliata
    df["any_incorrect"] = df["correct_windows"].apply(lambda lst: any(x is False for x in lst))

    # Salva file arricchito
    output_path = os.path.join(output_dir, file)
    df.to_csv(output_path, index=False)
    logger.info(f"Aggiornato con correct info: {file}")


#Definisco colonne d'interesse
signal_cols = [
    "Accel_LN_X", "Accel_LN_Y", "Accel_LN_Z",
    "Accel_WR_X", "Accel_WR_Y", "Accel_WR_Z",
    "Gyro_X", "Gyro_Y", "Gyro_Z"
]
plot_sensor_channels_with_colored_line_and_windows(
    r'C:\codes\HumanActivityRecognition\data\downstream_data\df_with_windows_with_correct\df_spoon_action_10.csv',
    signal_cols
)

plot_sensor_channels_with_colored_line_and_windows(
    r'C:\codes\HumanActivityRecognition\data\downstream_data\df_with_windows_with_correct\df_spoon_action_11.csv',
    signal_cols
)

plot_sensor_channels_with_colored_line_and_windows(
    r'C:\codes\HumanActivityRecognition\data\downstream_data\df_with_windows_with_correct\df_spoon_action_41.csv',
    signal_cols
)


df_spoon_non_null = pd.read_csv(r'C:\codes\HumanActivityRecognition\data\downstream_data\df_SPOON_non_null.csv')

# Carico e concatena tutti i file di interessa
correct_files = [
    f for f in os.listdir(r'C:\codes\HumanActivityRecognition\data\downstream_data\df_with_windows_with_correct')
    if f.startswith('df_spoon_action_') and f.endswith('.csv')
]
dfs_correct = []
for f in correct_files:
    df = pd.read_csv(os.path.join(r'C:\codes\HumanActivityRecognition\data\downstream_data\df_with_windows_with_correct', f))
    dfs_correct.append(df)
df_correct_all = pd.concat(dfs_correct, ignore_index=True)

# Colonne chiave per il merge
key_cols = [
    'Timestamp', 'Accel_LN_X', 'Accel_LN_Y', 'Accel_LN_Z',
    'Accel_WR_X', 'Accel_WR_Y', 'Accel_WR_Z',
    'Gyro_X', 'Gyro_Y', 'Gyro_Z'
]

# Colonne da aggiungere
extra_cols = [
    'window_ids', 'correct_windows', 'correct_summary', 'any_incorrect',
    # aggiungo qui tutte le colonne che vuoi portare da df_spoon a df_spoon_non_null
]


# Merge
df_merged = pd.merge(
    df_spoon_non_null,
    df_correct_all[key_cols + extra_cols],
    on=key_cols,
    how='left'
)

# Salvo il risultato
df_merged.to_csv(r'C:\codes\HumanActivityRecognition\data\downstream_data\df_SPOON_non_null_with_correctness.csv', index=False)



df_main = pd.read_csv(r'C:\codes\HumanActivityRecognition\data\downstream_data\3002_SP.csv')
df_correct = pd.read_csv(r'C:\codes\HumanActivityRecognition\data\downstream_data\df_SPOON_non_null_with_correctness.csv')

# Filtro solo le righe con kid_id == 3002
df_correct = df_correct[df_correct['kid_id'] == 3002]

# Merge left
df_merged = pd.merge(
    df_main,
    df_correct[key_cols + extra_cols],
    on=key_cols,
    how='left'
)

# Salvo il risultato
df_merged.to_csv(r'C:\codes\HumanActivityRecognition\data\downstream_data\3002_SP_with_correctness.csv', index=False)




plot_clean_activity_timeline_compress(
    r'C:\codes\HumanActivityRecognition\data\downstream_data\3002_SP_with_correctness.csv',
    signal_cols,
    min_activity_length=20,  # Solo segmenti di almeno 20
    pause_compression_factor=50,
    debug=True
)

radar_activity_duration(r'C:\codes\HumanActivityRecognition\data\downstream_data\3002_SP_with_correctness.csv', activity_col="action")