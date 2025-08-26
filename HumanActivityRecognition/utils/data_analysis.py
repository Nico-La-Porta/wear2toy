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
from typing import Dict, List, Any
from utils.log_config import logger

from scipy.stats import kstest, norm

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

    logger.info(f"Grafico salvato in: {file_path}")


   #Visualizza la distribuzione delle finestre in base alle etichette.
def plot_window_distribution(X, Y,dataset_type="train",save_dir="figures"):
    num_windows = len(Y)
    # Controllo delle dimensioni
    logger.info(f"Numero totale di finestre: {num_windows}")
    logger.info(f"Forma delle finestre (X): {X.shape}")
    logger.info(f"Numero di etichette uniche: {len(np.unique(Y))}")


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

    logger.info(f"Grafico salvato in: {file_path}")

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

    logger.info(f"File JSON salvato in: {json_file_path}")




def compute_descriptive_statistics(grouped_data: Dict[int, List[np.ndarray]]) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Calcola statistiche descrittive per ogni attività.

    :param grouped_data: Dizionario con attività come chiavi e liste di tracce come valori.
    :return: Dizionario con attività come chiavi e un dizionario di statistiche come valori.
    """
    stats = {}

    for activity, traces in grouped_data.items():
        logger.info(f"Calcolo statistiche descrittive per l'attività {activity} con {len(traces)} tracce")
        all_data = np.vstack(traces)  # Unisco tutte le tracce in un unico array per analisi

        # Escludo la prima colonna (timestamp)
        data_without_timestamp = all_data[:, 1:]  # Rimuovo la prima colonna (timestamp)
        # Calcolo acc_norm (radice della somma dei quadrati delle prime tre colonne)
        acc_norm = np.sqrt(np.sum(data_without_timestamp [:, :3] ** 2, axis=1))

        # Statistiche descrittive
        stats[activity] = {
            "mean": np.mean(data_without_timestamp, axis=0).tolist(),
            "std": np.std(data_without_timestamp, axis=0).tolist(),
            "min": np.min(data_without_timestamp , axis=0).tolist(),
            "max": np.max(data_without_timestamp , axis=0).tolist(),
            "acc_norm_mean": float(np.mean(acc_norm)), 
            "acc_norm_std": float(np.std(acc_norm)), 
            "acc_norm_min": float(np.min(acc_norm)),
            "acc_norm_max": float(np.max(acc_norm))
        }
        output_path = os.path.join(FIGURES_DIR, f"stats_activity_{activity}.json")
        with open(output_path, "w") as f:
            json.dump(stats[activity], f, indent=4)
        logger.info(f"Statistiche salvate in: {output_path}")

    return stats


# Funzione per tracciare un istogramma e fare il test KS
def plot_histogram_and_ks_test(data: np.ndarray, activity: int):
    """
    Traccia un istogramma e verifica se i dati seguono una distribuzione normale usando il KS test.
    
    :param data: Dati della traccia (un array NumPy).
    :param activity: Attività per cui si sta facendo il test.
    """
    # Traccio l'istogramma dei dati
    plt.figure(figsize=(8, 6))
    plt.hist(data.flatten(), bins='auto', color='blue', alpha=0.7, label=f'Activity {activity}')
    plt.title(f'Histogram of Data for Activity {activity}')
    plt.xlabel('Data Value')
    plt.ylabel('Frequency')
    plt.legend(loc='best')
    plt.show()

    # Eseguo il KS test confrontando con la distribuzione normale
    # Calcolo la media e la deviazione standard dei dati per adattarli alla normale
    mu, std = norm.fit(data.flatten())
    
    # Eseguo il KS test
    stat, p_value = kstest(data.flatten(), 'norm', args=(mu, std))
    print(f'Activity {activity} - p-value for KS test: {p_value}')

    if p_value < 0.05:
        print(f"Data for Activity {activity} do not follow a normal distribution (rejected by KS test).")
    else:
        print(f"Data for Activity {activity} follow a normal distribution (not rejected by KS test).")

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
    


