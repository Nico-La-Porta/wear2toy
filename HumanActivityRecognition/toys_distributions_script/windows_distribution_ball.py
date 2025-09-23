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
import logging
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
#definisco il path da cui leggere i .csv

from utils.log_config import logger
from figures import plot_CM, create_single_distribution_bar_chart
import normalization
from utils import mapping_activity
#definisco il path da cui leggere i .csv

path="C:\codes\HumanActivityRecognition\data\downstream_data"
logger.debug(path)



#Visto che l'informazione relativa all'id del bambino lo abbiamo, così come abbiamo anche l'informazione relativa
#al giocattolo, vado a filtrare i .csv in modo tale da avere solo le righe che hanno l'attività non nulla e poi
#appendo tutte le righe delle righe non nulle in un unico dataframe
#per tutti i file che terminano in .csv nella cartella path
# Definisco il percorso della cartella contenente i CSV

# Nome del file CSV finale
final_csv_path = os.path.join(path, 'df_BA_non_null.csv')

# Controllo se il file esiste già
if os.path.exists(final_csv_path):
    logger.debug(f"Il file {final_csv_path} esiste già. Lo sto caricando...")
    df_ball = pd.read_csv(final_csv_path)
else:
    #trovo tutti i file che corrispondono a "BA" nella cartella path e li stampo a schermo
    files = glob.glob(os.path.join(path, "*_BA*.csv"))
    logger.debug(f"Files: {files}")
    
    kid_ball, kid_ball_no_null = [], [] # liste per salvare utenti prima e dopo il merge 
    df_list_ball = [] # lista vuota per appendere i dataframe con attività non nulla

    for file in files:
        df = pd.read_csv(file)
        
        logger.debug(f"Original shape: {df.shape}")
        kid_ball.append(df['kid_id'].unique())
        df = df[df['action_id'] != 0] #filtro le righe con action_id non nullo
        logger.debug(f"Filtered shape: {df.shape}")
        kid_ball_no_null.append(df['kid_id'].unique())

        logger.debug(f"Columns: {df.columns}")
        logger.debug(f"Action counts:\n{df['action'].value_counts()}")
        logger.debug(f"Toy counts:\n{df['toy_id'].value_counts()}")
        logger.debug("="*50)  # Separatore tra i file

    # mi stampo gli utenti prima di fare il merge e dopo il merge
    if len(kid_ball) > 0:
        logger.info(f"Numero di utenti che hanno fatto almeno un azione: {len(kid_ball_no_null)/len(kid_ball)}")
    else:
        logger.info("Nessun utente trovato nei file analizzati (kid_ball è vuoto).")

    '''
    Visto che l'informazione relativa all'id del bambino lo abbiamo, così come abbiamo anche l'informazione relativa
    al giocattolo, vado a filtrare i .csv in modo tale da avere solo le righe che hanno l'attività non nulla e poi
    appendo tutte le righe non nulle in un unico dataframe per tutti i file che terminano in .csv nella balltella path
    '''

    for file in os.listdir(path):
        if not file.endswith('.csv'):
            continue
    
        #leggo solo i file che dopo l'undescore ha BA*.csv
        if file.split('_')[-1].startswith('BA') and file.endswith('.csv'): #controllo che il file termini con .csv
            df_temp=pd.read_csv(os.path.join(path,file))
            df_temp = df_temp[df_temp['action_id'] != 0]
            df_list_ball.append(df_temp)

    df_ball = pd.concat(df_list_ball)
    logger.info("Dimensioni del df_ball con tutte le attività non nulle")
    logger.info(df_ball.shape)
    logger.info(df_ball.columns)
    logger.info(df_ball['action'].value_counts())

    # salvo il dataframe
    df_ball.to_csv(os.path.join(path,'df_BA_non_null.csv'),index=False)


#ora divido il dataframe in base all'attività (action_id) e salvo i dataframe in un file .csv
#per ogni attività

for action_id in df_ball['action_id'].unique():
    df_action = df_ball[df_ball["action_id"] == action_id] #filtro il dataframe in base all'attività
    logger.debug(f"Dimensioni del dataframe df_ball_action_{action_id} - {df_action.shape}") #log delle dimensioni del dataframe
    logger.info(f"Conteggio delle attività per df_ball_action_{action_id}") #log del conteggio delle attività
    #salvo il dataframe
    df_action.to_csv(os.path.join(path,f'df_ball_action_{action_id}.csv'),index=False) #index=False per non salvare l'indice
    logger.debug(f"Salvato il dataframe df_ball_action_{action_id}.csv")


#applico sliding window con la funzion process_csv
nb_sensor_channels = 9
sliding_window_length = 100
sliding_window_step = 50

#ora applico la funzione sliding window (che mi da come output x_window e y_window) a tutti i .csv relativi al giocattolo ball
#e poi concateno tutto in un unica x e y 

X, Y = [], []
kid_action_counts = {}

#deve essere esattamaente del formato df_ball_action_{action_id}.csv e non deve ferminare con median_normalized o normalized_pt
for action_file in [
    f for f in os.listdir(path)
    if (
        f.endswith('.csv')
        and f.startswith('df_ball_action_')
        and not f.endswith('median_normalized.csv')
        and not f.endswith('normalized_pt.csv')
    )
]:
    file_path = os.path.join(path, action_file)
    action = action_file.split('_')[-1].split('.')[0]

    X_windows, Y_windows, kid_id_action_dict = sliding_window_on_data.process_csv(file_path, nb_sensor_channels, sliding_window_length, sliding_window_step)

    X.append(X_windows)
    Y.append(Y_windows)


    logger.info(f"Numero  totale di finestre per l'azione {action}:{len(X_windows)}")
    kid_action_counts[f"ball_action_{action}"] = kid_id_action_dict #aggiungo il dizionario al dizionario principale per tenere traccia del numero di finestre per ogni bambino per ogni azione, ogni ball_action è una chiave e il valore è un dizionario con il numero di finestre per ogni bambino
    logger.info(f"Contenuto finale di kid_action_counts: {kid_action_counts}") #per veere quante finestre per ogni azione e per ogni bambino sono state elaborte 


# Concateno tutti i dati in un unico array per X e Y
X = np.concatenate(X, axis=0)
Y = np.concatenate(Y, axis=0)


Y = Y.flatten()
Y_names = mapping_activity.convert_original_to_names(Y, "BALL")
# Crea il bar chart della distribuzione completa
create_single_distribution_bar_chart(
    Y=Y_names,
    toy_name="BALL",
    title_suffix="",
    save_path=os.path.join(FIGURES_DIR, "ball"),
    use_class_prefix=False  
)
