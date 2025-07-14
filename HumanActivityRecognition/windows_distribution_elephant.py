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
#definisco il path da cui leggere i .csv

from utils.log_config import logger
from figures import plot_CM, combine_kfold_confusion_matrices, create_single_distribution_bar_chart
import normalization
from utils import mapping_activity
from sklearn.model_selection import StratifiedKFold

#definisco il path da cui leggere i .csv

path="C:\codes\HumanActivityRecognition\data\downstream_data"
logger.debug(path)

mapping_activity.save_all_mappings()

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


#ora divido il dataframe in base all'attività (action_id) e salvo i dataframe in un file .csv
#per ogni attività

for action_id in df_elep_filtered['action_id'].unique():
    df_action = df_elep_filtered[df_elep_filtered["action_id"] == action_id] #filtro il dataframe in base all'attività
    logger.debug(f"Dimensioni del dataframe df_elep_filtered_action_{action_id} - {df_action.shape}") #log delle dimensioni del dataframe
    logger.info(f"Conteggio delle attività per df_elep_filtered_action_{action_id}") #log del conteggio delle attività
    #salvo il dataframe
    df_action.to_csv(os.path.join(path,f'df_elep_filtered_action_{action_id}.csv'),index=False) #index=False per non salvare l'indice
    logger.debug(f"Salvato il dataframe df_elep_filtered_action_{action_id}.csv")



#applico sliding window con la funzion process_csv
nb_sensor_channels = 9
sliding_window_length = 100
sliding_window_step = 50

#ora applico la funzione sliding window (che mi da come output x_window e y_window) a tutti i .csv relativi al giocattolo ball
#e poi concateno tutto in un unica x e y 

X, Y = [], []
kid_action_counts = {}

for action_file in [f for f in os.listdir(path) if f.endswith('.csv') and f.split('_')[1] == 'elep']:
    file_path = os.path.join(path, action_file)
    action = action_file.split('_')[-1].split('.')[0]

    X_windows, Y_windows, kid_id_action_dict = sliding_window_on_data.process_csv(file_path, nb_sensor_channels, sliding_window_length, sliding_window_step)

    X.append(X_windows)
    Y.append(Y_windows)


    logger.info(f"Numero  totale di finestre per l'azione {action}:{len(X_windows)}")
    kid_action_counts[f"elep_action_{action}"] = kid_id_action_dict #aggiungo il dizionario al dizionario principale per tenere traccia del numero di finestre per ogni bambino per ogni azione, ogni ball_action è una chiave e il valore è un dizionario con il numero di finestre per ogni bambino
    logger.info(f"Contenuto finale di kid_action_counts: {kid_action_counts}") #per veere quante finestre per ogni azione e per ogni bambino sono state elaborte 
    #totale numero di finestre indipendentemente dal bambino
    logger.info(f"Numero totale di finestre per l'azione {action}: {len(X_windows)}")


# Concateno tutti i dati in un unico array per X e Y
X = np.concatenate(X, axis=0)
Y = np.concatenate(Y, axis=0)


# Stampo le dimensioni di X e Y
logger.info(f"Dimensioni di X finale: {X.shape}")
logger.info(f"Dimensioni di Y finale: {Y.shape}")


Y = Y.flatten()
Y_names = mapping_activity.convert_original_to_names(Y, "ELEPHANT")
# Crea il bar chart della distribuzione completa
create_single_distribution_bar_chart(
    Y=Y_names,
    toy_name="ELEPHANT",
    title_suffix="",
    save_path=os.path.join(FIGURES_DIR, "elephant_complete_distribution"),
    use_class_prefix=False  
)

