import os
import torch
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from utils.log_config import logger
from glob import glob
from sklearn.metrics import classification_report



from collections import Counter





import matplotlib.patches as mpatches



from app_config import FIGURES_DIR, REPORTS_DIR
from models.DeepConvLSTM import DeepConvLSTM, HARDataset


def plot_CM(mdl_class, mdl_weights: str, X: np.ndarray, Y
: np.ndarray, batch_size: int, figure_name: str, labels_dict: dict = None):
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
        model = mdl_class(n_classes=n_classes, nb_sensor_channels=9, sliding_window_length=100)
        # Carico i psi NEL modello, non sull'OrderedDict
        state_dict = torch.load(mdl_weights, map_location=device)
        model.load_state_dict(state_dict, strict=False)  # strict=False per sicurezza
        # Sposto il MODELLO sul device, non lo state_dict
        model.to(device)
        model.eval() # Imposto il modello in modalità di valutazione e quindi non calcolo i gradienti e disabilito il dropout
        
        logger.info(f"Loaded complete model from {mdl_weights}")
      
       

 
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
 
        # Calcolo dell'F1-score
        f1_scores = {
            'macro': f1_score(all_labels, all_preds, average='macro'),
            'micro': f1_score(all_labels, all_preds, average='micro'),
            'weighted': f1_score(all_labels, all_preds, average='weighted')
        }
 
        # Per classe
        f1_per_class = f1_score(all_labels, all_preds, average=None)
 
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
 
        # Salvo l'F1-score in un file TXT
        # Rimuovo le prime due lettere dal nome della figura
        f1_filename = figure_name[2:] if len(figure_name) > 2 else figure_name
        f1_file_txt = os.path.join(REPORTS_DIR, f"f1-score_{f1_filename}.txt")
 
        with open(f1_file_txt, 'w') as f:
            f.write(f"F1-Score (Macro): {f1_scores['macro']:.6f}\n")
            f.write(f"F1-Score (Micro): {f1_scores['micro']:.6f}\n")
            f.write(f"F1-Score (Weighted): {f1_scores['weighted']:.6f}\n\n")
            f.write("F1-Score per classe:\n")
            for i, score in enumerate(f1_per_class):
                class_name = labels_dict[i] if labels_dict else f"Class {i}"
                f.write(f"{class_name}: {score:.6f}\n")
       
        logger.info(f"F1-scores saved to: {f1_file_txt}")
 
       
 
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
        plt.figure(figsize=(16, 14), dpi=300)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Greys",
            xticklabels=labels_names, yticklabels=labels_names,
            linewidths=0.3, square=True, annot_kws={"size": 10}, cbar=True)
 
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.yticks(rotation=0, fontsize=12)
        plt.title(f"Confusion Matrix - {figure_name}", fontsize=16)
        plt.xlabel("Predicted", fontsize=16)
        plt.ylabel("True", fontsize=16)
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
        import traceback
        logger.error(f"Traceback completo: {traceback.format_exc()}")
        return None, None



def combine_kfold_confusion_matrices(fold_results_dir, num_folds=3, toy_name="ball", save_path=None):
    """
    Combina le confusion matrix di tutti i fold in una singola CM finale
    
    Args:
        fold_results_dir: Directory contenente i CSV delle predizioni per ogni fold
        num_folds: Numero di fold utilizzati
        toy_name: Nome del giocattolo (es. "ball", "car", etc.)
        save_path: Path dove salvare la CM combinata
    """
    
    all_true_labels = []
    all_predicted_labels = []
    
    # Leggo i risultati di ogni fold
    for fold in range(1, num_folds + 1):
        # Pattern flessibile per i nomi dei file CSV
        predictions_file = os.path.join(fold_results_dir, f"predictions__test_fold_{fold}_kfold_inference_{toy_name}_{num_folds}_folds.csv")
        
        if os.path.exists(predictions_file):
            df_fold = pd.read_csv(predictions_file)
            
            #colonne sono true e predicted
            true_col = 'true' if 'true' in df_fold.columns else df_fold.columns[0]
            pred_col = 'predicted' if 'predicted' in df_fold.columns else df_fold.columns[1]
            
            all_true_labels.extend(df_fold[true_col].tolist())
            all_predicted_labels.extend(df_fold[pred_col].tolist())
            
            print(f"Fold {fold}: {len(df_fold)} predizioni caricate")
        else:
            print(f"WARNING: File {predictions_file} non trovato")
    
    # Controllo se sono stati caricati dati
    if len(all_true_labels) == 0:
        print("ERROR: Nessun dato caricato. Verifica i nomi dei file CSV.")
        return None, None, None
    
    # Converto in array numpy
    all_true_labels = np.array(all_true_labels)
    all_predicted_labels = np.array(all_predicted_labels)
    
    print(f"\nTotale predizioni: {len(all_true_labels)}")
    print(f"Classi uniche (true): {np.unique(all_true_labels)}")
    print(f"Classi uniche (predicted): {np.unique(all_predicted_labels)}")
    
    # Creo la confusion matrix combinata
    cm_combined = confusion_matrix(all_true_labels, all_predicted_labels)
    
    # Ottiengo  classi
    classes = np.unique(np.concatenate([all_true_labels, all_predicted_labels]))
    
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_combined, annot=True, fmt='d', cmap='Greys', 
                xticklabels=classes, yticklabels=classes,
                cbar_kws={'label': 'Count'})
    plt.title(f'Combined Confusion Matrix - {num_folds} Fold CV ({toy_name.upper()})')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix salvata in: {save_path}")
    
    plt.show()
    
    # Calcooa metriche
    accuracy = np.trace(cm_combined) / np.sum(cm_combined)
    
    # Report di classificazione
    report = classification_report(all_true_labels, all_predicted_labels, 
                                 target_names=[f'Class_{i}' for i in classes])
    
    print("\n" + "="*50)
    print("CONFUSION MATRIX COMBINATA")
    print("="*50)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"\nConfusion Matrix:\n{cm_combined}")
    print(f"\nClassification Report:\n{report}")
    
    # Salvo i risultati
    if save_path:
        base_path = os.path.splitext(save_path)[0]
        
        # Saloa la confusion matrix come CSV
        cm_df = pd.DataFrame(cm_combined, 
                           index=[f'True_{i}' for i in classes],
                           columns=[f'Pred_{i}' for i in classes])
        cm_df.to_csv(f"{base_path}_matrix.csv")
        
        # Salvo le predizioni combinate
        combined_predictions = pd.DataFrame({
            'true': all_true_labels,
            'predicted': all_predicted_labels
        })
        combined_predictions.to_csv(f"{base_path}_predictions.csv", index=False)
        
        # Salvo le metriche
        metrics = {
            'accuracy': accuracy,
            'total_predictions': len(all_true_labels),
            'num_folds': num_folds,
            'toy_name': toy_name
        }
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(f"{base_path}_metrics.csv", index=False)
        
        print(f"File salvati con prefisso: {base_path}")
    
    return cm_combined, all_true_labels, all_predicted_labels



# Configurazione dei parametri di stile per i grafici
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['grid.alpha'] = 0.3

def create_class_distribution_bar_chart(Y_train, Y_test, toy_name="CAR", save_path=None):
    """
    Crea un bar diagram publication-ready con tre grafici affiancati
    
    Parameters:
    - Y_train: array delle etichette del training set
    - Y_test: array delle etichette del test set
    - toy_name: nome del giocattolo (default: "CAR")
    - save_path: percorso per salvare il grafico (opzionale)
    """
    
    # Calcola le distribuzioni
    train_distribution = Counter(Y_train)
    test_distribution = Counter(Y_test)
    
    # Ottengo tutte le classi uniche
    all_classes = sorted(set(list(train_distribution.keys()) + list(test_distribution.keys())))
    
    # Calcolo i totali
    total_train = len(Y_train)
    total_test = len(Y_test)
    total_overall = total_train + total_test
    
    # Preparo i dati per il grafico
    train_counts = [train_distribution.get(cls, 0) for cls in all_classes] # ottengo i conteggi per ogni classe nel training set
    test_counts = [test_distribution.get(cls, 0) for cls in all_classes] # ottengo i conteggi per ogni classe nel test set
    total_counts = [train_counts[i] + test_counts[i] for i in range(len(all_classes))] # sommo i conteggi per ottenere la distribuzione complessiva
    
    # Calcolo le percentuali
    train_percentages = [(count / total_train) * 100 for count in train_counts] # calcolo le percentuali per il training set
    test_percentages = [(count / total_test) * 100 for count in test_counts] # calcolo le percentuali per il test set
    overall_percentages = [(count / total_overall) * 100 for count in total_counts] # calcolo le percentuali per la distribuzione complessiva
    
    # creo il grafico 
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    #COLORI
    colors = ['#2C3E50', '#E74C3C', '#3498DB', '#27AE60', '#F39C12', '#9B59B6', '#1ABC9C', '#34495E']
    bar_colors = [colors[i % len(colors)] for i in range(len(all_classes))]
    
    # Grafico 1: Training Set
    bars1 = ax1.bar(range(len(all_classes)), train_percentages, color=bar_colors, 
                    alpha=0.8, edgecolor='black', linewidth=0.8, hatch='///')
    ax1.set_xlabel('Activity Class', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax1.set_title(f'{toy_name} Dataset - Training Set', fontsize=13, fontweight='bold', pad=15)
    ax1.set_xticks(range(len(all_classes)))
    ax1.set_xticklabels([f'C{cls}' for cls in all_classes], fontsize=11)
    ax1.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
    ax1.set_axisbelow(True)
    
    # Rimuovi spines superiori e destri
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_linewidth(0.8)
    ax1.spines['bottom'].set_linewidth(0.8)
    
    # Aggiungi percentuali sopra le barre
    for i, (bar, percentage) in enumerate(zip(bars1, train_percentages)):
        if percentage > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{percentage:.1f}%', ha='center', va='bottom', 
                    fontsize=10, fontweight='bold')
    
    # Grafico 2: Test Set
    bars2 = ax2.bar(range(len(all_classes)), test_percentages, color=bar_colors, 
                    alpha=0.8, edgecolor='black', linewidth=0.8, hatch='...')
    ax2.set_xlabel('Activity Class', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax2.set_title(f'{toy_name} Dataset - Test Set', fontsize=13, fontweight='bold', pad=15)
    ax2.set_xticks(range(len(all_classes)))
    ax2.set_xticklabels([f'C{cls}' for cls in all_classes], fontsize=11)
    ax2.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
    ax2.set_axisbelow(True)
    
    # Rimuovi spines superiori e destri
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_linewidth(0.8)
    ax2.spines['bottom'].set_linewidth(0.8)
    
    # Aggiungi percentuali sopra le barre
    for i, (bar, percentage) in enumerate(zip(bars2, test_percentages)):
        if percentage > 0:
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{percentage:.1f}%', ha='center', va='bottom', 
                    fontsize=10, fontweight='bold')
    
    # Grafico 3: Distribuzione Complessiva
    bars3 = ax3.bar(range(len(all_classes)), overall_percentages, color=bar_colors, 
                    alpha=0.8, edgecolor='black', linewidth=0.8, hatch='xxx')
    ax3.set_xlabel('Activity Class', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax3.set_title(f'{toy_name} Dataset - Overall Distribution', fontsize=13, fontweight='bold', pad=15)
    ax3.set_xticks(range(len(all_classes)))
    ax3.set_xticklabels([f'C{cls}' for cls in all_classes], fontsize=11)
    ax3.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
    ax3.set_axisbelow(True)
    
    # Rimuovi spines superiori e destri
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['left'].set_linewidth(0.8)
    ax3.spines['bottom'].set_linewidth(0.8)
    
    # Aggiungi percentuali sopra le barre
    for i, (bar, percentage) in enumerate(zip(bars3, overall_percentages)):
        if percentage > 0:
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{percentage:.1f}%', ha='center', va='bottom', 
                    fontsize=10, fontweight='bold')
    
    # Layout professionale
    plt.tight_layout()
    
    # SalvO il grafico 
    if save_path:
        base_path = save_path.rsplit('.', 1)[0] if '.' in save_path else save_path
        
        # PDF per LaTeX 
        plt.savefig(f"{base_path}.pdf", dpi=300, bbox_inches='tight', 
                   format='pdf', facecolor='white', edgecolor='none')
        
        # PNG ad alta risoluzione
        plt.savefig(f"{base_path}.png", dpi=300, bbox_inches='tight', 
                   format='png', facecolor='white', edgecolor='none')
        
        print(f"Grafici salvati in: {base_path}.[pdf|png]")
    
    plt.show()
    
    # Stampo statistiche 
    print(f"\n=== {toy_name} Dataset Statistics ===")
    print(f"Training samples: {total_train}")
    print(f"Test samples: {total_test}")
    print(f"Total samples: {total_overall}")
    print(f"Number of classes: {len(all_classes)}")


def create_side_by_side_bar_chart(Y_train, Y_test, toy_name="CAR", save_path=None):
    """
    Crea un bar diagram side-by-side 
    
    Parameters:
    - Y_train: array delle etichette del training set
    - Y_test: array delle etichette del test set
    - toy_name: nome del giocattolo (default: "CAR")
    - save_path: percorso per salvare il grafico (opzionale)
    """
    
    # Calcola le distribuzioni
    train_distribution = Counter(Y_train)
    test_distribution = Counter(Y_test)
    
    # Ottieni tutte le classi uniche
    all_classes = sorted(set(list(train_distribution.keys()) + list(test_distribution.keys())))
    
    # Calcola i totali
    total_train = len(Y_train)
    total_test = len(Y_test)
    
    # Prepara i dati per il grafico
    train_counts = [train_distribution.get(cls, 0) for cls in all_classes]
    test_counts = [test_distribution.get(cls, 0) for cls in all_classes]
    
    # Calcola le percentuali
    train_percentages = [(count / total_train) * 100 for count in train_counts]
    test_percentages = [(count / total_test) * 100 for count in test_counts]
    
    # Crea il grafico con stile LNCS
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Posizioni delle barre
    x = np.arange(len(all_classes))
    width = 0.35
    
    # Colori LNCS-friendly
    train_color = '#2C3E50'  # Blu scuro
    test_color = '#E74C3C'   # Rosso
    
    # Crea le barre con pattern diversi
    bars1 = ax.bar(x - width/2, train_percentages, width, label='Training Set', 
                   color=train_color, alpha=0.8, edgecolor='black', linewidth=0.8, hatch='///')
    bars2 = ax.bar(x + width/2, test_percentages, width, label='Test Set', 
                   color=test_color, alpha=0.8, edgecolor='black', linewidth=0.8, hatch='...')
    
    # Personalizza il grafico per LNCS
    ax.set_xlabel('Activity Class', fontsize=14, fontweight='bold')
    ax.set_ylabel('Percentage (%)', fontsize=14, fontweight='bold')
    ax.set_title(f'{toy_name} Dataset - Class Distribution', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([f'C{cls}' for cls in all_classes], fontsize=12)
    ax.tick_params(axis='y', labelsize=12)
    
    # Griglia professionale
    ax.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Legenda professionale
    legend = ax.legend(loc='upper right', fontsize=12, frameon=True, 
                      fancybox=True, shadow=True, framealpha=0.95)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('black')
    legend.get_frame().set_linewidth(0.5)
    
    # Rimuovi spines superiori e destri
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    
    # Aggiungi percentuali sopra le barre
    for i, (bar, percentage) in enumerate(zip(bars1, train_percentages)):
        if percentage > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                   f'{percentage:.1f}%', ha='center', va='bottom', 
                   fontsize=10, fontweight='bold')
    
    for i, (bar, percentage) in enumerate(zip(bars2, test_percentages)):
        if percentage > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                   f'{percentage:.1f}%', ha='center', va='bottom', 
                   fontsize=10, fontweight='bold')
    
    # Layout professionale
    plt.tight_layout()
    
    # Salva il grafico in formato publication-ready
    if save_path:
        base_path = save_path.rsplit('.', 1)[0] if '.' in save_path else save_path
        
        # PDF per LaTeX (preferito per LNCS)
        plt.savefig(f"{base_path}.pdf", dpi=300, bbox_inches='tight', 
                   format='pdf', facecolor='white', edgecolor='none')
        
        # PNG ad alta risoluzione
        plt.savefig(f"{base_path}.png", dpi=300, bbox_inches='tight', 
                   format='png', facecolor='white', edgecolor='none')
        
        print(f"Grafici salvati in: {base_path}.[pdf|png]")
    
    plt.show()
    
    # Stampa statistiche per il paper
    print(f"\n=== {toy_name} Dataset Statistics ===")
    print(f"Training samples: {total_train}")
    print(f"Test samples: {total_test}")
    print(f"Total samples: {total_train + total_test}")
    print(f"Number of classes: {len(all_classes)}")
    
    # Stampa dettaglio per classe
    print("\nClass Details:")
    for i, cls in enumerate(all_classes):
        print(f"  Class C{cls}: Train={train_counts[i]} ({train_percentages[i]:.1f}%), "
              f"Test={test_counts[i]} ({test_percentages[i]:.1f}%)")


# Esempio di utilizzo
if __name__ == "__main__":
    print("Funzioni disponibili:")
    print("- create_class_distribution_bar_chart() - 3 grafici affiancati")
    print("- create_side_by_side_bar_chart() - grafico side-by-side")