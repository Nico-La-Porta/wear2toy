import os

import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn.functional as F

from app_config import MODELS_DIR, FIGURES_DIR, REPORTS_DIR
from utils.log_config import logger
from utils.plot import reliability_plot


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')
 
    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False"""


def calculate_class_weights(y_labels, num_classes):
    """
    Calcola i pesi delle classi per bilanciare il dataset
    """
    
    # Calcolo i pesi usando sklearn:  peso_classe_i = n_samples / (n_classes * n_samples_classe_i)
    class_weights = compute_class_weight( 
        'balanced',
        classes=np.arange(num_classes),
        y=y_labels
    )
    
    # Converte in tensor PyTorch
    class_weights_tensor = torch.FloatTensor(class_weights)
    
    logger.info(f"Pesi delle classi calcolati: {class_weights}")
    
    return class_weights_tensor

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_f1score = -float('inf')  # F1-score migliore inizializzato a -inf

    def early_stop(self, f1score):
        if f1score > self.best_f1score + self.min_delta:  # Se l'F1-score migliora
            self.best_f1score = f1score
            self.counter = 0  # Reset del contatore
        else:
            self.counter += 1  # Inizia a contare quando l'F1-score non migliora
            if self.counter >= self.patience:  # Se non migliora per 'patience' epoche
                return True  # Fermati
        return False


#Colleziono tutte le predizioni e le etichette dell'intera epoca e poi calcolo l'F1 score una sola volta su tutti i dati.

def train(net, train_loader, val_loader, 
          exp_figures_dir: str,
          exp_reports_dir: str,
          # Altri parametri
          epochs: int = 10,
          lr: float = 0.01,
          patience: int = 7,
          figure_name: str = "figure",
          f1_average: str = 'macro',
          criterion=None,
          # Parametro per controllare se salvare i grafici di questa sessione di training
          save_plots: bool = False):
    
    """
    Addestra e valida un modello di rete neurale.

    Args:
        net (torch.nn.Module): Il modello da addestrare.
        train_loader (DataLoader): DataLoader per il set di training.
        val_loader (DataLoader): DataLoader per il set di validazione.
        exp_figures_dir (str): Cartella dove salvare i grafici.
        exp_reports_dir (str): Cartella dove salvare i report.
        epochs (int): Numero massimo di epoche.
        lr (float): Learning rate.
        patience (int): Pazienza per l'early stopping.
        figure_name (str): Nome base per i file dei grafici.
        f1_average (str): Metodo di calcolo per F1-score multi-classe.
        criterion: La funzione di loss. Se None, usa CrossEntropyLoss.
        save_plots (bool): Se True, salva i grafici delle curve di apprendimento.
    
    Returns:
        float: Il miglior F1-score ottenuto sul set di validazione.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    # Imposta l'ottimizzatore e la funzione di loss.
    opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    # Usa il criterion passato come parametro, altrimenti usa CrossEntropyLoss di default
    if criterion is None:
        criterion = torch.nn.CrossEntropyLoss()

    criterion.to(device)
    

    # Inizializza le liste per salvare la cronologia delle performance.
    train_loss_history, val_loss_history = [], []
    train_f1_history, val_f1_history = [], []


    # Inizializza l'early stopper e la variabile per il miglior F1 score di validazione.
    early_stopper = EarlyStopper(patience=patience, min_delta=0.001)
    best_val_f1 = 0.0
    last_train_f1 = 0.0
    logger.info(f"--- Inizio Training per '{figure_name}' ({epochs} epoche) ---")

    # Ciclo di Training per ogni Epoca
    # ------------------------------------
    for e in range(epochs):
        
        # --- Fase di Training per l'Epoca Corrente ---
        net.train() # Mette il modello in modalità training (attiva dropout, etc.)
        train_losses = []
        # Liste temporanee per raccogliere le predizioni e le etichette dell'epoca.
        epoch_train_preds, epoch_train_labels = [], []

        for batch_data in train_loader:
            # Spacchetta i dati del batch. Ignoro i metadati (indici, etc.) con `_`.
            inputs, targets, _, _ = batch_data
            inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
            batch_size = inputs.size(0)
            
            # Azzera i gradienti, inizializza lo stato nascosto.
            opt.zero_grad()
            h = net.init_hidden(batch_size)
            
            # Forward pass: ottiene l'output del modello.
            output, h = net(inputs, h, batch_size)
            
            # Calcola la loss e fa la backpropagation.
            loss = criterion(output, targets.long())
            loss.backward()
            opt.step()
            
            # Salva la loss e le predizioni del batch.
            train_losses.append(loss.item())
            _, predicted = torch.max(output, 1)
            epoch_train_preds.extend(predicted.cpu().numpy())
            epoch_train_labels.extend(targets.cpu().numpy())
        
        # Calcola le metriche medie per l'intera epoca di training.
        train_loss_history.append(np.mean(train_losses))
        current_train_f1 = f1_score(epoch_train_labels, epoch_train_preds, average=f1_average, zero_division=0)
        train_f1_history.append(current_train_f1)
        last_train_f1 = current_train_f1 #salvo valore epoca corrente
        if val_loader:
        # --- Fase di Validazione per l'Epoca Corrente ---
            net.eval() # Mette il modello in modalità valutazione (disattiva dropout, etc.)
            val_losses = []
            epoch_val_preds, epoch_val_labels = [], []
            
            with torch.no_grad(): # Disabilita il calcolo dei gradienti per la validazione.
                for batch_data in val_loader:
                    inputs, targets, _, _ = batch_data
                    inputs, targets = inputs.to(device), targets.to(device)
                    batch_size = inputs.size(0)
                    val_h = net.init_hidden(batch_size)
                    output, _ = net(inputs, val_h, batch_size)
                    val_loss = criterion(output, targets.long())
                    val_losses.append(val_loss.item())
                    _, predicted = torch.max(output, 1)
                    epoch_val_preds.extend(predicted.cpu().numpy())
                    epoch_val_labels.extend(targets.cpu().numpy())

            # Calcola le metriche medie per l'intera epoca di validazione.
            val_loss_history.append(np.mean(val_losses))
            current_val_f1 = f1_score(epoch_val_labels, epoch_val_preds, average=f1_average, zero_division=0)
            val_f1_history.append(current_val_f1)
            
            # Aggiorna il miglior F1 score di validazione trovato finora.
            if current_val_f1 > best_val_f1:
                best_val_f1 = current_val_f1
                logger.info(f"Nuovo miglior F1-score di validazione: {best_val_f1:.4f}")

            logger.info(f"Epoch {e+1}/{epochs} | Train Loss: {np.mean(train_losses):.4f} | Train F1: {current_train_f1:.4f} | Val Loss: {np.mean(val_losses):.4f} | Val F1: {current_val_f1:.4f}")
            
            # Controlla se è il caso di fermare il training in anticipo.
            if early_stopper.early_stop(current_val_f1):
                logger.info(f"Early stopping attivato all'epoca {e+1} perché non ci sono miglioramenti.")
                break
        else:
            logger.info(f"Epoch {e+1}/{epochs} | Train Loss: {np.mean(train_losses):.4f} | Train F1: {current_train_f1:.4f} | (Nessuna validazione)")

    # Fine del Training: Salvataggio Grafici e Return
    # --------------------------------------------------
    # Se specificato, salva i grafici delle curve di apprendimento.
    if save_plots and val_loader:
        logger.info(f"Salvataggio dei grafici di training per '{figure_name}'...")
        os.makedirs(exp_figures_dir, exist_ok=True)
        
        # Grafico F1-Score
        plt.figure(figsize=(10, 6))
        plt.plot(train_f1_history, label="Train F1-Score")
        plt.plot(val_f1_history, label="Validation F1-Score")
        plt.title(f"F1 Score Curve - {figure_name}")
        plt.xlabel("Epoch")
        plt.ylabel(f"F1-Score ({f1_average})")
        plt.legend()
        plt.grid(True)
        f1_curve_path = os.path.join(exp_figures_dir, f"f1_curve_{figure_name}.png")
        plt.savefig(f1_curve_path, dpi=300)
        plt.close()
        
        # Grafico Loss
        plt.figure(figsize=(10, 6))
        plt.plot(train_loss_history, label="Train Loss")
        plt.plot(val_loss_history, label="Validation Loss")
        plt.title(f"Loss Curve - {figure_name}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        loss_curve_path = os.path.join(exp_figures_dir, f"loss_curve_{figure_name}.png")
        plt.savefig(loss_curve_path, dpi=300)
        plt.close()
        
        logger.info(f"Grafici di training salvati in: {exp_figures_dir}")

    # Ritorna il miglior F1 score di validazione. Questo valore è cruciale per Optuna.
    logger.info(f"Training per '{figure_name}' completato. Miglior F1-score di validazione raggiunto: {best_val_f1:.4f}")
    # Decide quale valore restituire in base alla presenza del val_loader.
    if val_loader:
        # Se c'era validazione, restituisco il miglior F1 score di validazione.
        # Questo è il valore che serve a Optuna per l'ottimizzazione.
        logger.info(f"Training con validazione completato. Miglior F1 di validazione: {best_val_f1:.4f}")
        return best_val_f1
    else:
        # Se non c'era validazione (training finale), restituiamo l'F1 score
        # di training dell'ultima epoca. Questo ci dà una metrica concreta.
        logger.info(f"Training finale completato. F1 di training dell'ultima epoca: {last_train_f1:.4f}")
        return last_train_f1



def evaluate_model(net, test_loader, exp_figures_dir: str, 
                   exp_reports_dir: str,
                   figure_name="evaluation", 
                   f1_average='macro', 
                   save_confusion_matrix: bool = False, 
                   save_f1_score: bool = True, 
                   save_predictions_csv: bool = True, 
                   criterion=None, 
                   labels_dict=None,
                   plot_calibration_curve: bool = False):
    net.eval()
    if criterion is None:
        criterion = torch.nn.CrossEntropyLoss()
    all_test_preds = []
    all_test_labels = []
    all_test_metadata = []
    all_test_consecutivity = []
    val_losses = []
    val_accuracy = 0

    # DEBUG: Verifica dimensioni del DataLoader
    logger.info(f"\n=== DEBUG EVALUATE_MODEL ===")
    logger.info(f"Test loader dataset size: {len(test_loader.dataset)}")
    logger.info(f"Test loader number of batches: {len(test_loader)}")
    logger.info(f"Expected total samples: {len(test_loader.dataset)}")

    total_samples_processed = 0  # DEBUG: Conta finestre processate


    with torch.no_grad():
        for batch_idx, batch_data in enumerate(test_loader):
            logger.info(f"Processing batch {batch_idx + 1}/{len(test_loader)}")
            if len(batch_data) == 4:  # data, labels, indices, consecutivity
                inputs, targets, indices, consecutivity = batch_data
            else:
                inputs, targets = batch_data
                indices = list(range(len(targets)))
                consecutivity = [True] * len(targets)
                
            batch_size = inputs.size(0)
            logger.info(f"Batch {batch_idx + 1}: {batch_size} finestre")
            val_h = net.init_hidden(batch_size)
            val_h = tuple([each.data for each in val_h])
            net.to(device)

            inputs, targets = inputs.to(device), targets.to(device)

            output, val_h = net(inputs, val_h, batch_size)
            loss = criterion(output, targets.long())
            val_losses.append(loss.item())

            _, predicted = torch.max(output, 1)
            all_test_preds.extend(predicted.cpu().numpy())
            all_test_labels.extend(targets.cpu().numpy())

            # CALIBRATION - Collect confidences for reliability plot
            probabilities = F.softmax(output, dim=1)  # [BS, N_CLASSES]
            max_conf, _ = torch.max(probabilities, dim=1)
            if 'all_test_confs' not in locals():
                all_test_confs = []
            all_test_confs.extend(max_conf.cpu().numpy())

            top_p, top_class = output.topk(1, dim=1)
            equals = predicted == targets.long()
            val_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()


            all_test_metadata.extend(indices)
            all_test_consecutivity.extend(consecutivity)
            logger.info(f"Batch {batch_idx + 1} - Indici salvati: {indices}")

        # CALIBRATION - Plot the reliability_diagram
        if 'all_test_confs' in locals() and len(all_test_confs) > 0:
            reliability_plot(all_test_confs,
                             all_test_preds,
                             all_test_labels,
                             save_path=exp_reports_dir)

        # DEBUG: Risultati finali
    logger.info(f"Totale finestre processate: {total_samples_processed}")
    logger.info(f"Lunghezza liste finali:")
    logger.info(f"  - all_test_preds: {len(all_test_preds)}")
    logger.info(f"  - all_test_labels: {len(all_test_labels)}")
    logger.info(f"  - all_test_metadata: {len(all_test_metadata)}")
    logger.info(f"  - all_test_consecutivity: {len(all_test_consecutivity)}")
    # Calcolo del mean_loss e mean_accuracy 
    mean_loss = np.mean(val_losses)
    mean_accuracy = val_accuracy / len(test_loader)
    
    # Calcolo dell'F1 score con diverse strategie
    f1_scores = {
        'macro': f1_score(all_test_labels, all_test_preds, average='macro'),
        'micro': f1_score(all_test_labels, all_test_preds, average='micro'),
        'weighted': f1_score(all_test_labels, all_test_preds, average='weighted')
    }
    
    # F1-score per classe
    f1_per_class = f1_score(all_test_labels, all_test_preds, average=None)

    logger.info(f"Evaluation Results - Loss: {mean_loss:.4f}, Accuracy: {mean_accuracy:.4f}, F1-Score ({f1_average}): {f1_scores[f1_average]:.4f}")
    
    # DataFrame con tutte le informazioni necessarie per l'analisi degli errori
    predictions_df = pd.DataFrame({
            # Estraiamo l'ID globale da ogni dizionario di metadati
            'global_window_id': [meta['global_window_id'] for meta in all_test_metadata],
            'true_label': all_test_labels,
            'predicted_label': all_test_preds,
            'is_consecutive': all_test_consecutivity,
            'correct': [true == pred for true, pred in zip(all_test_labels, all_test_preds)]
        })


    # Aggiungi nomi delle azioni se il mapping è fornito
    if labels_dict:
        predictions_df['true_action'] = [labels_dict.get(label, f'Unknown_{label}') for label in all_test_labels]
        predictions_df['predicted_action'] = [labels_dict.get(label, f'Unknown_{label}') for label in all_test_preds]


    # Salvataggio del CSV con informazioni complete
    if save_predictions_csv:
        os.makedirs(exp_reports_dir, exist_ok=True)
        pred_path = os.path.join(exp_reports_dir, f"predictions_{figure_name}.csv")
        predictions_df.to_csv(pred_path, index=False)
        logger.info(f"Test predictions saved to: {pred_path}")


    # Salvataggio F1-score
    if save_f1_score:
        
        os.makedirs(exp_reports_dir, exist_ok=True)
        
         # Salvo il file TXT con un nome coerente.
        f1_file_txt = os.path.join(exp_reports_dir, f"f1-score_{figure_name}.txt")
        
        # Salvo l'F1-score in TXT
        with open(f1_file_txt, 'w') as f:
            f.write(f"F1-Score (Macro): {f1_scores['macro']:.6f}\n")
            f.write(f"F1-Score (Micro): {f1_scores['micro']:.6f}\n")
            f.write(f"F1-Score (Weighted): {f1_scores['weighted']:.6f}\n\n")
            f.write("F1-Score per classe:\n")
            
            # Verifica se le classi sono disponibili nel dataset
            class_names = getattr(test_loader.dataset, 'classes', None)
            
            for i, score in enumerate(f1_per_class):
                class_name = class_names[i] if class_names else f"Class {i}"
                f.write(f"{class_name}: {score:.6f}\n")
        
        logger.info(f"F1-scores saved to: {f1_file_txt}")

    if save_confusion_matrix:
        os.makedirs(exp_figures_dir, exist_ok=True)
        cm = confusion_matrix(all_test_labels, all_test_preds)
        plt.figure(figsize=(16, 14), dpi=300)
        
        # Determina le etichette per gli assi
        if labels_dict:
            # Usa i nomi delle azioni
            tick_labels = [labels_dict.get(i, f'Class_{i}') for i in range(len(np.unique(all_test_labels + all_test_preds)))]
        else:
            # Fallback ai nomi del dataset
            tick_labels = getattr(test_loader.dataset, 'classes', [f'Class_{i}' for i in range(cm.shape[0])])
        
        sns.heatmap(cm, annot=True, fmt="d", cmap="Greys",  # in scala di grigi
                    xticklabels=tick_labels,
                    yticklabels=tick_labels,
                    linewidths=0.3, square=True, annot_kws={"size": 10}, cbar=True)
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.yticks(rotation=0, fontsize=12)
        plt.title(f"Confusion Matrix - {figure_name}", fontsize=16) 
        plt.xlabel("Predicted", fontsize=16)
        plt.ylabel("True", fontsize=16)
        plt.tight_layout()  
        cm_path = os.path.join(exp_figures_dir, f"cm_{figure_name}.png")
        plt.savefig(cm_path, bbox_inches='tight', dpi=300)
        plt.close()
        logger.info(f"Confusion Matrix saved to: {cm_path}")


    return mean_loss, mean_accuracy, f1_scores[f1_average]
