import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from datetime import datetime
from app_config import RAW_DATA_DIR_TRAIN, RAW_DATA_DIR_TEST
from utils import data_preprocessing
from run_config import SLIDING_WINDOW_LENGTH
from run_config import NB_SENSOR_CHANNELS
from run_config import SLIDING_WINDOW_STEP
import sliding_window_on_data
from app_config import FIGURES_DIR
import json


# Funzione per calcolare e rappresentare la distribuzione delle etichette
def plot_label_distribution(dataset,dataset_type="train",save_dir="figures"):
    # Estrai le etichette da tutte le tracce
    labels = np.concatenate([trace['TraceData'][:, -1] for trace in dataset])
    
    # Assicurati che sia un array 1D
    labels = labels.astype(int).flatten()

    # Ottieni tutte le etichette uniche
    unique_labels = np.unique(labels)

    # Crea il grafico
    plt.figure(figsize=(12, 6))
    sns.histplot(labels, bins=len(unique_labels), kde=False, color="skyblue")
    plt.title(f"Distribuzione delle etichette ({dataset_type})")
    plt.xlabel("Etichetta")
    plt.ylabel("Frequenza")

    # Mostra tutti i valori unici sull'asse x
    plt.xticks(ticks=unique_labels, labels=unique_labels, rotation=45)
    plt.tight_layout()
    #plt.show()

        # Salva il grafico nella cartella specificata
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"label_distribution_{dataset_type}_{timestamp}.png"
    file_path = os.path.join(save_dir, file_name)
    
    plt.savefig(file_path)
    plt.close()  # Chiude la figura per liberare la memoria

    print(f"Grafico salvato in: {file_path}")


   #Visualizza la distribuzione delle finestre in base alle etichette.
def plot_window_distribution(X, Y,dataset_type="train",save_dir="figures"):
    num_windows = len(Y)
    # Controllo delle dimensioni
    print(f"Numero totale di finestre: {num_windows}")
    print(f"Forma delle finestre (X): {X.shape}")
    print(f"Numero di etichette uniche: {len(np.unique(Y))}")

    # Ottieni tutte le etichette uniche
    unique_labels,counts = np.unique(Y,return_counts=True)

    # Creazione del grafico
    plt.figure(figsize=(12, 6))
    sns.histplot(Y, bins=len(unique_labels), kde=False, color="lightblue")
    plt.title(f"Distribuzione delle finestre per etichetta ({dataset_type})")
    plt.xlabel("Etichetta")
    plt.ylabel("Numero di finestre")

    # Mostra tutte le etichette sull'asse x
    plt.xticks(ticks=unique_labels, labels=unique_labels, rotation=45)
    plt.tight_layout()
    #plt.show()

        # Salva il grafico nella cartella specificata
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"window_distribution_{dataset_type}_{timestamp}.png"
    file_path = os.path.join(save_dir, file_name)
    
    plt.savefig(file_path)
    plt.close()  # Chiude la figura per liberare la memoria

    print(f"Grafico salvato in: {file_path}")

        # Salva il numero di finestre in un file JSON
    # Salva il numero di finestre in un file JSON
    json_data = {
        "dataset_type": dataset_type,
        "num_windows": num_windows,
        "windows_per_activity": dict(zip(unique_labels.tolist(),counts.tolist()))
    }
    
    json_file_name = f"window_distribution_{dataset_type}_{timestamp}.json"
    json_file_path = os.path.join(save_dir, json_file_name)
    
    with open(json_file_path, 'w') as json_file:
        json.dump(json_data, json_file,indent=4)

    print(f"Informazioni salvate in: {json_file_path}")


if __name__ == "__main__":
    # Esempio di utilizzo delle funzioni
    # Prepara i dati
    datasetTracesTrain = data_preprocessing.build_dataset(RAW_DATA_DIR_TRAIN)
    datasetTracesTest = data_preprocessing.build_dataset(RAW_DATA_DIR_TEST)
    dataset_train_labled = data_preprocessing.add_labels_to_dataset(datasetTracesTrain)
    dataset_test_labled = data_preprocessing.add_labels_to_dataset(datasetTracesTest)
    X_Train, Y_Train = sliding_window_on_data.apply_sliding_window(dataset_train_labled, SLIDING_WINDOW_LENGTH, SLIDING_WINDOW_STEP, NB_SENSOR_CHANNELS)
    X_Test, Y_Test = sliding_window_on_data.apply_sliding_window(dataset_test_labled, SLIDING_WINDOW_LENGTH, SLIDING_WINDOW_STEP, NB_SENSOR_CHANNELS)

    # Concatenare tutte le finestre e le etichette
    X_all = np.concatenate((X_Train, X_Test), axis=0)
    Y_all = np.concatenate((Y_Train, Y_Test), axis=0)
    plot_window_distribution(X_all, Y_all, dataset_type="all dataset",save_dir=FIGURES_DIR)
    plot_window_distribution(X_Train, Y_Train, dataset_type="train",save_dir=FIGURES_DIR)
    plot_window_distribution(X_Test,Y_Test, dataset_type="test",save_dir=FIGURES_DIR)
    


