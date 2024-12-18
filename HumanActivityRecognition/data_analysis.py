import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys


from app_config import PROJ_ROOT, RAW_DATA_DIR_TRAIN, RAW_DATA_DIR_TEST

from utils import data_preprocessing
from run_config import SLIDING_WINDOW_LENGTH
from run_config import NB_SENSOR_CHANNELS
from run_config import SLIDING_WINDOW_STEP
import sliding_window_on_data


# Funzione per calcolare e rappresentare la distribuzione delle etichette
def plot_label_distribution(dataset):
    # Estrai le etichette da tutte le tracce
    labels = np.concatenate([trace['TraceData'][:, -1] for trace in dataset])
    
    # Assicurati che sia un array 1D
    labels = labels.astype(int).flatten()

    # Ottieni tutte le etichette uniche
    unique_labels = np.unique(labels)

    # Crea il grafico
    plt.figure(figsize=(12, 6))
    sns.histplot(labels, bins=len(unique_labels), kde=False, color="skyblue")
    plt.title("Distribuzione delle etichette")
    plt.xlabel("Etichetta")
    plt.ylabel("Frequenza")

    # Mostra tutti i valori unici sull'asse x
    plt.xticks(ticks=unique_labels, labels=unique_labels, rotation=45)
    plt.tight_layout()
    plt.show()


   #Visualizza la distribuzione delle finestre in base alle etichette.
def plot_window_distribution(X, Y):

    # Controllo delle dimensioni
    print(f"Numero totale di finestre: {len(Y)}")
    print(f"Forma delle finestre (X): {X.shape}")
    print(f"Numero di etichette uniche: {len(np.unique(Y))}")

    # Ottieni tutte le etichette uniche
    unique_labels = np.unique(Y)

    # Creazione del grafico
    plt.figure(figsize=(12, 6))
    sns.histplot(Y, bins=len(unique_labels), kde=False, color="lightblue")
    plt.title("Distribuzione delle finestre per etichetta")
    plt.xlabel("Etichetta")
    plt.ylabel("Numero di finestre")

    # Mostra tutte le etichette sull'asse x
    plt.xticks(ticks=unique_labels, labels=unique_labels, rotation=45)
    plt.tight_layout()
    plt.show()

# Visualizza la distribuzione delle finestre in base alle etichette
def plot_window_distribution(X, Y):
    # Controllo delle dimensioni
    print(f"Numero totale di finestre: {len(Y)}")
    print(f"Forma delle finestre (X): {X.shape}")
    print(f"Numero di etichette uniche: {len(np.unique(Y))}")

    # Ottieni tutte le etichette uniche
    unique_labels = np.unique(Y)

    # Creazione del grafico
    plt.figure(figsize=(12, 6))
    sns.histplot(Y, bins=len(unique_labels), kde=False, color="lightblue")
    plt.title("Distribuzione delle finestre per etichetta")
    plt.xlabel("Etichetta")
    plt.ylabel("Numero di finestre")

    # Mostra tutte le etichette sull'asse x
    plt.xticks(ticks=unique_labels, labels=unique_labels, rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Esempio di utilizzo delle funzioni
    # Prepara i dati
    datasetTracesTrain = data_preprocessing.build_dataset(RAW_DATA_DIR_TRAIN)
    datasetTracesTest = data_preprocessing.build_dataset(RAW_DATA_DIR_TEST)
    dataset_train_labled = data_preprocessing.add_labels_to_dataset(datasetTracesTrain)
    dataset_test_labled = data_preprocessing.add_labels_to_dataset(datasetTracesTest)
    X_Train, Y_Train = sliding_window_on_data.apply_sliding_window(dataset_train_labled, SLIDING_WINDOW_LENGTH, SLIDING_WINDOW_STEP, NB_SENSOR_CHANNELS)
    X_Test, Y_Test = sliding_window_on_data.apply_sliding_window(dataset_test_labled, SLIDING_WINDOW_LENGTH, SLIDING_WINDOW_STEP, NB_SENSOR_CHANNELS)

    plot_window_distribution(X_Train, Y_Train)
    plot_window_distribution(X_Test,Y_Test)

