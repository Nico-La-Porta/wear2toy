import numpy as np
import os
import sys
import pandas as pd
from app_config import PROJ_ROOT
sys.path.append(os.path.join(PROJ_ROOT, "HumanActivityRecognition"))
from utils import data_processing
from utils.log_config import logger

# Funzione per applicare la sliding window sui dati di train
#prende in ingresso la lista di dizionari con TraceId e TraceData, la lunghezza della finestra, l'overlap (passo della finestra), e il numero di canali (feature)
#mi restituisce X_Train e Y_train (input e lable in formato utile per poi fare il train)
def apply_sliding_window(new_dataset_labeled, sliding_window_length, sliding_window_step, nb_sensor_channels):

    X= []
    Y= []

    for trace in new_dataset_labeled:
        data = trace['TraceData']

        # Separazione dei dati
        X_trace = data[:, 1:-2]  # Escludi prima colonna, penultima e ultima
        Y_trace = data[:, -1]   # Prendi l'ultima colonna (label)

        logger.info(f"Processing trace with shape: {data.shape}")
        logger.info(f"X_trace shape: {X_trace.shape}")
        logger.info(f"Y_trace shape: {Y_trace.shape}")

        # Applicazione della sliding window
        X_windows, padding_codes_x = data_processing.sliding_window(X_trace, ws=(sliding_window_length, X_trace.shape[1]), ss=(sliding_window_step, X_trace.shape[1]))

        logger.debug(f"X_windows shape after sliding window: {X_windows.shape}")
        logger.debug(f"Padding codes: {padding_codes_x}")

        Y_windows_full, padding_codes_y = data_processing.sliding_window(Y_trace.reshape(-1, 1), ws=(sliding_window_length, 1), ss=(sliding_window_step, 1))
        logger.debug(f"Y_windows_full shape after sliding window: {Y_windows_full.shape}")
        logger.debug(f"Padding codes: {padding_codes_y}")
        # Estrai l'ultima etichetta di ogni finestra
        Y_windows = np.asarray([window[np.nonzero(window)[0][0]] if np.any(window != 0) else 0 for window in Y_windows_full])
        logger.debug(f"Y_windows shape after extraction: {Y_windows.shape}")

        # Aggiungi le finestre ottenute a X_train e Y_train
        X.append(X_windows)
        Y.append(Y_windows)

    for i, window in enumerate(Y):
        logger.debug(f"Shape of Y window {i}: {window.shape}")

    Y = [window.flatten() for window in Y]
    # Conversione X_train e Y_train in numpy array per facilità di elaborazione successiva
    X = np.vstack(X)  # Concatenazione di tutte le finestre di X
    Y = np.concatenate(Y)  # Concatenazione di tutte le finestre di Y

    # Riscalatura dei dati (Conversione nei formati desiderati)

    X = X.astype(np.float32)
    Y = Y.reshape(len(Y)).astype(np.uint8)
    # Reshape finale per Conv1D
    X = X.reshape((-1, sliding_window_length, nb_sensor_channels))

    return X, Y


def process_csv(file_path, nb_sensor_channels, sliding_window_length, sliding_window_step):
    df = pd.read_csv(file_path)
    logger.debug(f"sto leggendo il file csv: {file_path}")
    #logger.debug("Colonne nel dataset:", df.columns.tolist())

    kid_ids = df['kid_id'].unique() # Prendo gli id dei bambini e li metto in una lista
    print(">>> Kid_ids:", kid_ids)
    X, Y, = [], [] 

    kid_id_action_count = {} # Dizionario per tenere traccia del numero di finestre per ogni kid

    for kid_id in kid_ids:
        X_kid = df[df['kid_id'] == kid_id].drop(columns=['Timestamp', 'Mag_X', 'Mag_Y', 'Mag_Z','Date_time', 'action', 'action_id', 'G',
                              'communication', 'social_interaction', 'restricted_repetitive_behaviour',
                              'ados_total_score', 'I', 'E', 'toy_id','kid_id']).to_numpy() #
        Y_kid = df[df['kid_id'] == kid_id]['action_id'].to_numpy() #

        #stampa di debug per vedere quanti dati ho per ogni kid
        logger.info(f"Kid_id: {kid_id}, X_kid shape: {X_kid.shape}, Y_kid shape: {Y_kid.shape}")

        # Applicazione della sliding window
        X_windows, padding_codes_x = data_processing.sliding_window(X_kid, ws=(sliding_window_length, X_kid.shape[1]), ss=(sliding_window_step, X_kid.shape[1]), min_pad_samples=30, extreme_pad_samples=10)
        logger.debug(f"X_windows shape after sliding window: {X_windows.shape}")
        logger.debug(f"Padding codes: {padding_codes_x}")


        Y_windows_full, padding_codes_y = data_processing.sliding_window(Y_kid.reshape(-1, 1), ws=(sliding_window_length, 1), ss=(sliding_window_step, 1), min_pad_samples=30, extreme_pad_samples=10)
        logger.debug(f"Y_windows_full shape after sliding window: {Y_windows_full.shape}")
        logger.debug(f"Padding codes: {padding_codes_y}")
        # Estrai l'ultima etichetta di ogni finestra
        Y_windows = np.asarray([window[np.nonzero(window)[0][0]] if np.any(window != 0) else 0 for window in Y_windows_full])
        logger.debug(f"Y_windows shape after extraction: {Y_windows.shape}")
        
        # Reshape per Conv1D
        #X_windows = X_windows.reshape((-1, sliding_window_length, nb_sensor_channels))



        X.append(X_windows)
        Y.append(Y_windows)


        kid_id_action_count[int(kid_id)] = len(Y_windows) # Salva il numero di finestre per ogni kid

        #stampo il numero di finestre che ha ogni azione e stampo quali azioni per ogni kid
        for action_id in np.unique(Y_windows):
            action_count = len(Y_windows[Y_windows == action_id])
            logger.info(f"Kid_id: {kid_id}, Action_id: {action_id}, Action_count: {action_count}")
    
    logger.debug(f"Conteggio finale di finestre per ogni bambino per l'azione {action_id}: {kid_id_action_count}")
    # Concateno le finestre di tutti i bambini
    X_all = np.concatenate(X, axis=0)  # Concateno le finestre per X per tutti i bambini
    Y_all = np.concatenate(Y, axis=0)  # Concateno le finestre per Y per tutti i bambini


    
    return X_all, Y_all, kid_id_action_count



def old_process_csv(file_path, nb_sensor_channels, sliding_window_length, sliding_window_step):
    df = pd.read_csv(file_path)
    logger.debug(f"sto leggendo il file csv: {file_path}")
    
    # Rimozione delle colonne non necessarie
    X_data = df.drop(columns=['Timestamp', 'Mag_X', 'Mag_Y', 'Mag_Z','Date_time', 'action', 'action_id', 'G',
                              'communication', 'social_interaction', 'restricted_repetitive_behaviour',
                              'ados_total_score', 'I', 'E', 'toy_id','kid_id'])
    
    logger.debug("Colonne nel dataset:", X_data.columns.tolist())
    X_data = X_data.to_numpy()
    logger.debug("Shape di X_data:", X_data.shape)
    
    # Estrarre le etichette
    Y_data = df['action_id'].to_numpy()
    logger.debug("Shape di Y_data:", Y_data.shape)
    
    # Applicazione della sliding window
    X_windows = data_processing.old_sliding_window(X_data, ws=(sliding_window_length, X_data.shape[1]),
                               ss=(sliding_window_step, X_data.shape[1]))
    Y_windows_full = data_processing.sliding_window(Y_data.reshape(-1, 1), ws=(sliding_window_length, 1), ss=(sliding_window_step, 1))
    
    # Prendere l'ultima etichetta di ogni finestra
    Y_windows = np.asarray([window[-1] for window in Y_windows_full])
    
    # Reshape per Conv1D
    X_windows = X_windows.reshape((-1, sliding_window_length, nb_sensor_channels))
    
    logger.debug("Shape di X_windows dopo reshape:", X_windows.shape)
    logger.debug("Shape di Y_windows dopo reshape:", Y_windows.shape)
    
    return X_windows, Y_windows

