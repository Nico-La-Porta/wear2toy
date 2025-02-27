import numpy as np
import os
import sys
import pandas as pd
from app_config import PROJ_ROOT
sys.path.append(os.path.join(PROJ_ROOT, "HumanActivityRecognition"))
from utils import data_processing

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

        print("X_trace shape:", X_trace.shape)
        print("Y_trace shape:", Y_trace.shape)

        # Applicazione della sliding window
        X_windows = data_processing.sliding_window(X_trace, ws=(sliding_window_length, X_trace.shape[1]), ss=(sliding_window_step, X_trace.shape[1]))
        print("X_windows shape:", X_windows.shape)

        Y_windows_full = data_processing.sliding_window(Y_trace.reshape(-1, 1), ws=(sliding_window_length, 1), ss=(sliding_window_step, 1))

        # Estrai l'ultima etichetta di ogni finestra
        Y_windows = np.asarray([window[-1] for window in Y_windows_full])
        print("Y_window shape:", Y_windows.shape)

        # Aggiungi le finestre ottenute a X_train e Y_train
        X.append(X_windows)
        Y.append(Y_windows)

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
    print("Colonne nel dataset:", df.columns.tolist())
    
    # Rimozione delle colonne non necessarie
    X_data = df.drop(columns=['Timestamp', 'Date_time', 'action', 'action_id', 'G',
                              'communication', 'social_interaction', 'restricted_repetitive_behaviour',
                              'ados_total_score', 'I', 'E', 'toy_id'])
    
    print("Feature selezionate:", X_data.columns.tolist())
    X_data = X_data.to_numpy()
    print("Shape di X_data:", X_data.shape)
    
    # Estrarre le etichette
    Y_data = df['action_id'].to_numpy()
    print("Shape di Y_data:", Y_data.shape)
    
    # Applicazione della sliding window
    X_windows = data_processing.sliding_window(X_data, ws=(sliding_window_length, X_data.shape[1]),
                               ss=(sliding_window_step, X_data.shape[1]))
    Y_windows_full = data_processing.sliding_window(Y_data.reshape(-1, 1), ws=(sliding_window_length, 1), ss=(sliding_window_step, 1))
    
    # Prendere l'ultima etichetta di ogni finestra
    Y_windows = np.asarray([window[-1] for window in Y_windows_full])
    
    # Reshape per Conv1D
    X_windows = X_windows.reshape((-1, sliding_window_length, nb_sensor_channels))
    
    print("Shape di X_windows:", X_windows.shape)
    print("Shape di Y_windows:", Y_windows.shape)
    
    return X_windows, Y_windows