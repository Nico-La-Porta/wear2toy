import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

