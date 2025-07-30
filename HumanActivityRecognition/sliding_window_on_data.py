import os
import sys
import numpy as np
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
    kid_ids_all= []  # Lista per tenere traccia di tutti i kid_id
    is_consecutive = []
    window_row_indices = []  #per tracciare gli indici delle righe originali
    kid_id_action_count = {} # Dizionario per tenere traccia del numero di finestre per ogni kid
    global_window_id = 0

    for kid_id in kid_ids:
        kid_mask = df['kid_id'] == kid_id
        kid_df = df[kid_mask].reset_index(drop=True)
        X_kid = df[df['kid_id'] == kid_id].drop(columns=['Timestamp', 'Accel_WR_X' ,'Accel_WR_Y' ,'Accel_WR_Z' ,'Date_time', 'action', 'action_id', 'G',
                              'communication', 'social_interaction', 'restricted_repetitive_behaviour',
                              'ados_total_score', 'I', 'E', 'toy_id','kid_id']).to_numpy() #
        Y_kid = df[df['kid_id'] == kid_id]['action_id'].to_numpy() 

        # Estraggo i timestamp per questo bambino
        timestamps_kid = df[df['kid_id'] == kid_id]['Timestamp'].to_numpy()
        #stampa di debug per vedere quanti dati ho per ogni kid
        logger.info(f"Kid_id: {kid_id}, X_kid shape: {X_kid.shape}, Y_kid shape: {Y_kid.shape}")


        # --- SLIDING WINDOW SUGLI INDICI ---
        row_indices = np.arange(len(kid_df))  # Indici delle righe originali per questo kid
        row_indices = row_indices.reshape(-1, 1)  # shape (n_samples, 1)
        #Applicazione della sliding window sugli indici delle righe in modo da ottenere le finestre di indici
        row_windows, _ = data_processing.sliding_window(
            row_indices, ws=(sliding_window_length, 1), ss=(sliding_window_step, 1),
            min_pad_samples=30, extreme_pad_samples=10
        )
        row_windows = row_windows.squeeze(axis=2) if row_windows.ndim == 3 else row_windows  # shape (n_windows, window_length)

        # Trovo gli indici delle righe che sono 0 e li sostituisco con -1, tranne la prima finestra
        for i in range(row_windows.shape[0]):
            for j in range(row_windows.shape[1]):
                if row_windows[i, j] == 0:
                    # Solo la PRIMA finestra (i==0) può avere 0 come indice valido
                    if i != 0:
                        row_windows[i, j] = -1





        # Applicazione della sliding window
        X_windows, padding_codes_x= data_processing.sliding_window(X_kid, ws=(sliding_window_length, X_kid.shape[1]), ss=(sliding_window_step, X_kid.shape[1]), min_pad_samples=30, extreme_pad_samples=10)
        logger.debug(f"X_windows shape after sliding window: {X_windows.shape}")
        logger.debug(f"Padding codes for X: {padding_codes_x}")
        #prendo la prima etichetta non nulla di ogni finestra
        Y_windows_full, padding_codes_y = data_processing.sliding_window(Y_kid.reshape(-1, 1), ws=(sliding_window_length, 1), ss=(sliding_window_step, 1), min_pad_samples=30, extreme_pad_samples=10)
        logger.debug(f"Y_windows_full shape after sliding window: {Y_windows_full.shape}")
        logger.debug(f"Padding codes for Y: {padding_codes_y}")
        # Estraggo l'ultima etichetta di ogni finestra
        Y_windows = np.asarray([window[np.nonzero(window)[0][0]] if np.any(window != 0) else 0 for window in Y_windows_full])
        logger.debug(f"Y_windows shape after extraction: {Y_windows.shape}")


        # Calcolo la consecutività per ogni finestra basata sui timestamp
        consecutivity_info = calculate_window_consecutivity(
            timestamps_kid, sliding_window_length, sliding_window_step, len(X_windows)
        )
        
        # Salvo le informazioni per ogni finestra
        for i in range(len(X_windows)):
            start_idx = i * sliding_window_step
            end_idx = min(start_idx + sliding_window_length, len(kid_df))
            
            # Determino se la finestra è consecutiva
            is_consec = consecutivity_info[i]


            
            is_consecutive.append(is_consec)
            global_window_id += 1

        # Reshape per Conv1D
        #X_windows = X_windows.reshape((-1, sliding_window_length, nb_sensor_channels))



        X.append(X_windows)
        Y.append(Y_windows)
        kid_id_action_count[int(kid_id)] = len(Y_windows)
        kid_ids_all.extend([kid_id] * len(Y_windows))
        window_row_indices.extend(row_windows.tolist())  # Aggiungo gli indici delle righe per ogni finestra
        #stampo il numero di finestre che ha ogni azione e stampo quali azioni per ogni kid
        for action_id in np.unique(Y_windows):
            action_count = len(Y_windows[Y_windows == action_id])
            logger.info(f"Kid_id: {kid_id}, Action_id: {action_id}, Action_count: {action_count}")
    
    logger.debug(f"Conteggio finale di finestre per ogni bambino per l'azione {action_id}: {kid_id_action_count}")
    # Concateno le finestre di tutti i bambini
    X_all = np.concatenate(X, axis=0)  # Concateno le finestre per X per tutti i bambini
    Y_all = np.concatenate(Y, axis=0)  # Concateno le finestre per Y per tutti i bambini
    kid_ids_all = np.array(kid_ids_all)  # Converti in array numpy
    is_consecutive = np.array(is_consecutive)
    window_row_indices = np.array(window_row_indices)  # shape (n_windows, window_length)
    #stampo informazione relativa agli indici delle righe originali
    logger.debug(f"Shape of window_row_indices: {window_row_indices.shape}")
    
    return X_all, Y_all, kid_id_action_count,  is_consecutive ,  window_row_indices

def calculate_window_consecutivity(timestamps, window_length, window_step, num_windows):
    """
    Calcola se ogni finestra contiene timestamp consecutivi.
    
    Args:
        timestamps: array dei timestamp ordinati
        window_length: lunghezza della finestra
        window_step: passo della finestra
        num_windows: numero di finestre generate
    
    Returns:
        list: lista di booleani che indica se ogni finestra è consecutiva
    """
    consecutivity = [] #lista per tenere traccia della consecutività delle finestre
    
    # Calcolo la differenza media tra timestamp consecutivi => la differenza tra ogni coppia consecutiva di timestamp (esempio: [1, 2, 3] => [1, 1])
    time_diffs = np.diff(timestamps)
    median_diff = np.median(time_diffs) #calcolo la mediana delle differenze tra timestamp consecutivi (stima della differenza tipica tra timestamp consecutivi)
    
    # Soglia per considerare un gap troppo grande (es. 3 volte la differenza mediana), se la differenza tra due timestamp è maggiore di 3 volte la mediana, viene considerato un "buco" anomalo (cioè non consecutivo).
    gap_threshold = median_diff * 3
    
    #itero su tutte le finestre
    for i in range(num_windows):
        start_idx = i * window_step # Calcolo l'indice di inizio della finestra
        end_idx = min(start_idx + window_length, len(timestamps)) # Calcolo l'indice di fine della finestra
        
        if end_idx <= start_idx + 1:
            # Finestra troppo piccola (ma non capita mai)
            consecutivity.append(False)
            continue
            
        # Estraggo i timestamp della finestra
        window_timestamps = timestamps[start_idx:end_idx]
        
        # Calcolo le differenze tra timestamp consecutivi nella finestra
        window_diffs = np.diff(window_timestamps)
        
        # Verifico se ci sono gap troppo grandi rispetto alla soglia calcolata
        has_large_gaps = np.any(window_diffs > gap_threshold)
        
        # La finestra è consecutiva se non ha gap grandi
        is_consec = not has_large_gaps #Se non ci sono grandi gap, is_consec è True (finestra consecutiva)
        consecutivity.append(is_consec) #Restituisce la lista di booleani, uno per ogni finestra, che indica se i timestamp nella finestra erano consecutivi (True) o no (False)
        
    return consecutivity

def old_process_csv(file_path, nb_sensor_channels, sliding_window_length, sliding_window_step):
    df = pd.read_csv(file_path)
    print("Colonne nel dataset:", df.columns.tolist())
    
    # Rimozione delle colonne non necessarie
    X_data = df.drop(columns=['Timestamp', 'Mag_X', 'Mag_Y', 'Mag_Z','Date_time', 'action', 'action_id', 'G',
                              'communication', 'social_interaction', 'restricted_repetitive_behaviour',
                              'ados_total_score', 'I', 'E', 'toy_id','kid_id'])
    
    print("Feature selezionate:", X_data.columns.tolist())
    X_data = X_data.to_numpy()
    print("Shape di X_data:", X_data.shape)
    
    # Estrarre le etichette
    Y_data = df['action_id'].to_numpy()
    print("Shape di Y_data:", Y_data.shape)
    
    # Applicazione della sliding window
    X_windows = data_processing.old_sliding_window(X_data, ws=(sliding_window_length, X_data.shape[1]),
                               ss=(sliding_window_step, X_data.shape[1]))
    Y_windows_full = data_processing.sliding_window(Y_data.reshape(-1, 1), ws=(sliding_window_length, 1), ss=(sliding_window_step, 1))
    
    # Prendere l'ultima etichetta di ogni finestra
    Y_windows = np.asarray([window[-1] for window in Y_windows_full])
    
    # Reshape per Conv1D
    X_windows = X_windows.reshape((-1, sliding_window_length, nb_sensor_channels))
    
    print("Shape di X_windows:", X_windows.shape)
    print("Shape di Y_windows:", Y_windows.shape)
    
    return X_windows, Y_windows

