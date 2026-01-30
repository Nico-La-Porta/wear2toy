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
from glob import glob
from sklearn.metrics import classification_report
from collections import Counter
import matplotlib.patches as mpatches


from src.utils.log_config import logger
from models.DeepConvLSTM import DeepConvLSTM, HARDataset

# Configurazione dei parametri di stile per i grafici
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['grid.alpha'] = 0.3
def plot_CM(model, dataloader, device, figure_name: str, labels_dict: dict, 
                    exp_figures_dir: str, exp_reports_dir: str):
    """
    Funzione che accetta modello e un dataloader già pronti.
    Salva i risultati (matrice, F1, predizioni) nelle cartelle specifiche dell'esperimento.

    Parametri:
        model (torch.nn.Module): Il modello PyTorch GIÀ addestrato e pronto.
        dataloader (DataLoader): Il DataLoader contenente i dati da valutare.
        device (torch.device): Il device su cui eseguire il modello (es. "cuda" o "cpu").
        figure_name (str): Nome base per i file di output (es. "cm_train_fold_1").
        labels_dict (dict): Mappatura degli indici delle etichette ai nomi delle classi.
        exp_figures_dir (str): Percorso alla cartella delle figure per l'esperimento corrente.
        exp_reports_dir (str): Percorso alla cartella dei report per l'esperimento corrente.
    """
    logger.info(f"--- Avvio valutazione e plotting per: {figure_name} ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
    try:
        # Il modello viene passato già pronto e messo in modalità valutazione
        model.to(device)
        model.eval()
      
        # Dataset e DataLoader gia pronti 
        # Predizioni
        all_preds = [] #per memorizzare le predizioni
        all_labels = [] #per memorizzare le etichette vere
 
        with torch.no_grad():
            for data, labels, _, _ in dataloader:
                # Verifico che data e labels siano tensori PyTorch
                if not isinstance(data, torch.Tensor):
                    data = torch.tensor(data)
                if not isinstance(labels, torch.Tensor):
                    labels = torch.tensor(labels)
               
                data = data.to(device)
                batch_size = data.size(0)
                # Le etichette possono rimanere su CPU poiché non partecipano al calcolo del forward
                batch_len = len(data)
                hidden = model.init_hidden(batch_size=batch_size)  # Inizializzo lo stato nascosto per il batch corrente
                outputs, _ = model(data, hidden, batch_size)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                _, preds = torch.max(outputs, 1)  # Ottengo le predizioni massime (classi) per ogni campione nel batch  
 
               
                # Verifico che tutto sia su CPU prima di convertire in numpy ed aggiungo le predizioni e le etichette alla lista
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())  
 
        # Calcolo dell'F1-score
        f1_scores = {
            'macro': f1_score(all_labels, all_preds, average='macro', zero_division=0),
            'micro': f1_score(all_labels, all_preds, average='micro', zero_division=0),
            'weighted': f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        }
        f1_per_class = f1_score(all_labels, all_preds, average=None, zero_division=0)
 
        # Salvataggio delle predizioni in un file CSV
        os.makedirs(exp_reports_dir, exist_ok=True)  # Assicura che la directory per i risultati esista
        predictions_file = os.path.join(exp_reports_dir, f"{figure_name}_predictions.csv")
        predictions_df = pd.DataFrame({'true_label': all_labels, 'predicted_label': all_preds})
       
        # Se abbiamo un dizionario delle etichette, aggiungiamo anche i nomi delle classi
        if labels_dict:
            predictions_df['true_class'] = predictions_df['true_label'].map(labels_dict)
            predictions_df['predicted_class'] = predictions_df['predicted_label'].map(labels_dict)
        predictions_df.to_csv(predictions_file, index=False)
        logger.info(f"Predictions saved to: {predictions_file}")
 
        # Salvo l'F1-score in un file TXT
        # Rimuovo le prime due lettere dal nome della figura
        f1_filename = figure_name[2:] if len(figure_name) > 2 else figure_name
        f1_file_txt = os.path.join(exp_reports_dir, f"f1-score_{f1_filename}.txt")
 
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
        unique_labels = np.unique(np.concatenate((all_labels, all_preds)))
        labels_names = [labels_dict.get(i, str(i)) for i in unique_labels]
 
        # Etichette
        """if labels_dict:
            labels_names = [labels_dict[i] for i in range(len(cm))]
        else:
            labels_names = [str(i) for i in range(len(cm))]"""
 
        # Verifico che la directory di output esista
        os.makedirs(exp_figures_dir, exist_ok=True)
 
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
        plt.savefig(os.path.join(exp_figures_dir, f"{figure_name}.png"),
                    bbox_inches='tight', dpi=300)
        plt.close()
        logger.info(f"Confusion Matrix saved to: {os.path.join(exp_figures_dir, f'{figure_name}.png')}")
       
        return cm, predictions_df  # Restituisco sia la matrice di confusione che le predizioni

    except Exception as e:
        logger.error(f"Si è verificato un errore durante la creazione della matrice di confusione per {figure_name}: {e}")
        import traceback
        logger.error(f"Traceback completo: {traceback.format_exc()}")
        return None, None



def combine_kfold_confusion_matrices(exp_reports_dir, num_folds, class_names, exp_figures_dir):
    """
    Combina i risultati delle predizioni di tutti i fold per creare una CM aggregata.

    Parametri:
        exp_reports_dir (str): La cartella dei report dell'esperimento dove sono i file CSV.
        num_folds (int): Il numero di fold eseguiti.
        class_names (list): Lista dei nomi delle classi per il grafico.
        exp_figures_dir (str): La cartella dove salvare il grafico finale.
    """
    
    logger.info("\n--- Aggregazione delle Matrici di Confusione dei K-Fold ---")
    
    all_true_labels = []
    all_predicted_labels = []
    
    
    # Cicla su ogni fold per caricare i risultati
    for fold in range(1, num_folds + 1):
        # Il nome del file da cercare
        predictions_file = os.path.join(exp_reports_dir, f"predictions_TEST_fold_{fold}.csv")
        
        if os.path.exists(predictions_file):
            logger.info(f"Caricamento predizioni dal file: {predictions_file}")
            df_fold = pd.read_csv(predictions_file)
            all_true_labels.extend(df_fold['true_label'].tolist())
            all_predicted_labels.extend(df_fold['predicted_label'].tolist())
        else:
            logger.warning(f"File delle predizioni per il fold {fold} non trovato: {predictions_file}")


    if not all_true_labels:
        logger.error("Nessun file di predizioni trovato. Impossibile creare la matrice aggregata.")
        return None
    
   # Crea la confusion matrix combinata
    cm_combined = confusion_matrix(all_true_labels, all_predicted_labels)
    
    # Crea e salva il grafico
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_combined, annot=True, fmt='d', cmap='Greys', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Combined Confusion Matrix ({num_folds}-Fold CV)', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=16)
    plt.ylabel('True Label', fontsize=16)
    plt.tight_layout()

    save_path = os.path.join(exp_figures_dir, "cm_AGGREGATED_kfold.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    logger.info(f"Matrice di confusione aggregata salvata in: {save_path}")

    return cm_combined


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

def create_single_distribution_bar_chart(Y, toy_name="Dataset", save_path=None, title_suffix=" ", use_class_prefix=True):
    """
    Crea un bar diagram publication-ready per una singola distribuzione
    
    Parameters:
    - Y: array delle etichette
    - toy_name: nome del dataset/giocattolo 
    - save_path: percorso per salvare il grafico (opzionale)
    - title_suffix: suffisso per il titolo (es. "Complete Dataset", "Training Set", etc.)
    """
    
    # Calcola la distribuzione
    distribution = Counter(Y)
    
    # Ottieni tutte le classi uniche ordinate
    all_classes = sorted(distribution.keys())
    
    # Calcola il totale
    total_samples = len(Y)
    
    # Prepara i dati per il grafico
    counts = [distribution[cls] for cls in all_classes]
    percentages = [(count / total_samples) * 100 for count in counts]
    
    # Crea il grafico con stile LNCS
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Colori professionali
    colors = ['#2C3E50', '#E74C3C', '#3498DB', '#27AE60', '#F39C12', '#9B59B6', '#1ABC9C', '#34495E']
    bar_colors = [colors[i % len(colors)] for i in range(len(all_classes))]
    
    # Crea le barre
    bars = ax.bar(range(len(all_classes)), percentages, color=bar_colors, 
                  alpha=0.8, edgecolor='black', linewidth=0.8, hatch='///')
    
    # Personalizza il grafico per LNCS
    ax.set_xlabel('Activity Class', fontsize=16, fontweight='bold')
    ax.set_ylabel('Percentage (%)', fontsize=16, fontweight='bold')
    ax.set_title(f'{toy_name}  {title_suffix}', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(range(len(all_classes)))
    if use_class_prefix and all(isinstance(cls, (int, float)) for cls in all_classes):
        # Se sono numeri, aggiungi "C"
        ax.set_xticklabels([f'C{cls}' for cls in all_classes], fontsize=12)
    else:
        # Se sono stringhe (nomi azioni), usali direttamente
        ax.set_xticklabels(all_classes, fontsize=12, rotation=45, ha='right')
    ax.tick_params(axis='y', labelsize=12)
    
    # Griglia professionale
    ax.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Rimuovi spines superiori e destri
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    
    # Aggiungi percentuali sopra le barre
    for i, (bar, percentage, count) in enumerate(zip(bars, percentages, counts)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
               f'{percentage:.1f}%\n({count})', ha='center', va='bottom', 
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
        
        print(f"Grafico salvato in: {base_path}.[pdf|png]")

         # SVG vettoriale
        plt.savefig(f"{base_path}.svg", bbox_inches='tight', 
                   format='svg', facecolor='white', edgecolor='none')
        
        print(f"Grafico salvato in: {base_path}.[pdf|png|svg]")
    
    plt.show()
    
    # Stampa statistiche per il paper
    print(f"\n=== {toy_name} - {title_suffix} Statistics ===")
    print(f"Total samples: {total_samples}")
    print(f"Number of classes: {len(all_classes)}")
    
    # Stampa dettaglio per classe
    print("\nClass Details:")
    for i, cls in enumerate(all_classes):
        print(f"  Class C{cls}: {counts[i]} samples ({percentages[i]:.1f}%)")
    
    # Identifica classi problematiche
    min_count = min(counts)
    max_count = max(counts)
    if min_count < 10:
        rare_classes = [all_classes[i] for i, count in enumerate(counts) if count < 10]
        print(f"\nWARNING: Classes with <10 samples: {rare_classes}")
    
    if max_count / min_count > 10:
        print(f"WARNING: High class imbalance detected! Ratio: {max_count/min_count:.2f}")
    
    return distribution


# Esempio di utilizzo
if __name__ == "__main__":
    print("Funzioni disponibili:")
    print("- create_class_distribution_bar_chart() - 3 grafici affiancati")
    print("- create_side_by_side_bar_chart() - grafico side-by-side")