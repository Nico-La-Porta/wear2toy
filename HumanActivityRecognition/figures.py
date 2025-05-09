import os
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from utils.log_config import logger



from app_config import FIGURES_DIR, REPORTS_DIR
from models.DeepConvLSTM import DeepConvLSTM, HARDataset


def plot_CM(mdl_class, mdl_weights: str, X: np.ndarray, Y: np.ndarray, batch_size: int, figure_name: str, labels_dict: dict = None):
    """
    Mostra e salva la matrice di confusione per il modello e il dataset forniti.
    Inoltre salva le predizioni e le etichette reali in un file CSV.

    Parametri:
        mdl_class: La classe del modello (non un'istanza).
        mdl_weights (str): Percorso ai pesi del modello (file .pt).
        X (np.ndarray): Caratteristiche in input.
        Y (np.ndarray): Etichette reali (ground truth).
        batch_size (int): Dimensione del batch da usare nel DataLoader.
        figure_name (str): Nome del file della figura da salvare.
        labels_dict (dict, opzionale): Mappatura degli indici delle etichette ai nomi delle classi.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        # Carico il modello e i pesi
        n_classes = len(np.unique(Y)) #calcolo numero di classi andando a vedere le etichette uniche in y (etichette vere)
        logger.debug(f"Number of classes: {n_classes}")
        model = mdl_class(n_classes=n_classes) # Inizializzo il modello con il numero di classi corretto: creo un'istanza del modello
        model.load_state_dict(torch.load(mdl_weights, map_location=device)) # Carico i pesi del modello
        model.to(device)
        model.eval() # Imposto il modello in modalità di valutazione e quindi non calcolo i gradienti e disabilito il dropout

        # Dataset e DataLoader
        dataset = HARDataset(X, Y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        # Predizioni
        all_preds = [] #per memorizzare le predizioni
        all_labels = [] #per memorizzare le etichette vere

        with torch.no_grad():
            for data, labels in dataloader:
                # Verifico che data e labels siano tensori PyTorch
                if not isinstance(data, torch.Tensor):
                    data = torch.tensor(data)
                if not isinstance(labels, torch.Tensor):
                    labels = torch.tensor(labels)
                
                data = data.to(device)
                # Le etichette possono rimanere su CPU poiché non partecipano al calcolo del forward
                batch_len = len(data)
                hidden = model.init_hidden(batch_size=batch_len)  # Inizializzo lo stato nascosto per il batch corrente
                outputs = model(data, hidden, batch_len)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                _, preds = torch.max(outputs, 1)  # Ottengo le predizioni massime (classi) per ogni campione nel batch  

                
                # Verifico che tutto sia su CPU prima di convertire in numpy ed aggiungo le predizioni e le etichette alla lista 
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())  

        # Salvataggio delle predizioni in un file CSV
        os.makedirs(REPORTS_DIR, exist_ok=True)  # Assicura che la directory per i risultati esista
        predictions_file = os.path.join(REPORTS_DIR, f"{figure_name}_predictions.csv")
        
        # Creo un DataFrame con le etichette vere e le predizioni
        predictions_df = pd.DataFrame({
            'true_label': all_labels,
            'predicted_label': all_preds
        })
        
        # Se abbiamo un dizionario delle etichette, aggiungiamo anche i nomi delle classi
        if labels_dict:
            predictions_df['true_class'] = predictions_df['true_label'].map(labels_dict)
            predictions_df['predicted_class'] = predictions_df['predicted_label'].map(labels_dict)
        
        # Salvo il file CSV
        predictions_df.to_csv(predictions_file, index=False)
        logger.info(f"Predictions saved to: {predictions_file}")

        # Matrice di confusione
        cm = confusion_matrix(all_labels, all_preds)

        # Etichette
        if labels_dict:
            labels_names = [labels_dict[i] for i in range(len(cm))]
        else:
            labels_names = [str(i) for i in range(len(cm))]

        # Verifico che la directory di output esista
        os.makedirs(FIGURES_DIR, exist_ok=True)

        # Plot
        plt.figure(figsize=(10, 8), dpi=300)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Greys", xticklabels=labels_names, yticklabels=labels_names, linewidths=0.5, square=True, annot_kws={"size": 14})
        plt.title(f"Confusion Matrix - {figure_name}", fontsize=16) 
        plt.xlabel("Predicted", fontsize=16)
        plt.ylabel("True", fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.tight_layout()  

        # Salvataggio
        plt.savefig(os.path.join(FIGURES_DIR, f"{figure_name}.png"), 
                    bbox_inches='tight', dpi=300)  # Salvataggio in alta qualità
        plt.close()
        logger.info(f"Confusion Matrix saved")
        
        return cm, predictions_df  # Restituisco sia la matrice di confusione che le predizioni
    except FileNotFoundError:
        logger.error(f"File dei pesi non trovato: {mdl_weights}")
        return None, None
    except Exception as e:
        logger.error(f"Si è verificato un errore durante la creazione della matrice di confusione: {str(e)}")
        return None, None
