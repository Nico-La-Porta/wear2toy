from email import parser
import json
import os
import sys
import argparse
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import torch
import pickle
import random 
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
from collections import defaultdict


# Import da Optuna
import optuna
from optuna.pruners import MedianPruner

# Import custom modules
from app_config import REPORTS_DIR, FIGURES_DIR, MODELS_DIR
import src.utils.sliding_window_on_data as sliding_window_on_data
from models.DeepConvLSTM import DeepConvLSTM, HARDataset, collate_fn
import train_with_cm
import normalization
from utils.log_config import setup_logging, logger
from utils.focal_loss import FocalLoss, LabelSmoothingCrossEntropy, WeightedCrossEntropyLoss, CombinedLoss
from utils.figures import combine_kfold_confusion_matrices
from utils import mapping_activity
from utils.transformations import *
from utils.transformations_utils import *
from utils.data_loader import load_and_preprocess_data_unified, add_consecutive_segment_id

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
    
    #plt.show()
    
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
    
    return fig

def log_and_plot_distribution(Y, title_prefix, toy_name, toy_mapping, save_dir):
    """
    Prepara i dati e chiama la funzione di plotting per la distribuzione delle classi,
    utilizzando la funzione di conversione specializzata del progetto.
    """
    logger.info(f"===== ANALISI DISTRIBUZIONE: {title_prefix} - {toy_name.upper()} =====")

    if len(Y) == 0:
        logger.warning("L'array Y è vuoto. Impossibile mostrare la distribuzione.")
        return

    # --- INIZIO MODIFICA CHIAVE ---
    # Invece di fare una ricerca manuale nel dizionario, usiamo la tua funzione che già funziona!
    # Questa funzione gestirà correttamente qualsiasi problema di tipo (float vs int) o altre logiche.
    try:
        logger.info("Conversione degli ID numerici in nomi di azioni tramite 'convert_original_to_names'...")
        Y_names = mapping_activity.convert_original_to_names(Y, toy_name.upper())
    except Exception as e:
        logger.error(f"ERRORE: La funzione 'convert_original_to_names' ha fallito: {e}")
        logger.error("Ritorno al metodo di fallback (potrebbe mostrare degli 'ID_...')")
        # Fallback nel caso in cui la funzione non esista o dia errore
        labels_dict = toy_mapping["encoded_to_name"]
        Y_names = [labels_dict.get(int(y), f"ID_{y}") for y in Y]
    # --- FINE MODIFICA CHIAVE ---
    
    # Crea un nome di file pulito
    file_suffix = title_prefix.lower().replace(" ", "_").replace("(", "").replace(")", "")
    save_path_base = os.path.join(save_dir, f"{toy_name}_distribuzione_{file_suffix}")

    # Chiama la tua funzione di plotting (questa parte non cambia)
    fig = create_single_distribution_bar_chart(
        Y=Y_names,
        toy_name=toy_name.upper(),
        title_suffix=f"({title_prefix})",
        save_path=save_path_base,
        use_class_prefix=False
    )
    
    # Chiudi la figura per evitare che rimanga aperta in memoria
    if fig:
        plt.close(fig)


def get_pretrained_model_path(pt_norm, pt_aug):
    """
    Restituisce il path del modello pre-addestrato corretto in base alla configurazione.
    """
    #base_path = PRETRAINED_MODELS_DIR
    base_path = r'C:\codes\HumanActivityRecognition\models'
    
    if pt_norm == 'meanstd' and pt_aug == 'alltrs':
        return os.path.join(base_path, 'best_model_dl_meanstd_aug_sampler.pkl')
    elif pt_norm == 'iqr' and pt_aug == 'alltrs':
        return os.path.join(base_path, 'best_model_dl_iqr_aug.pkl')
    elif pt_norm == 'meanstd' and pt_aug == 'none':
        return os.path.join(base_path, 'best_model_dl_meanstd.pkl')
    elif pt_norm == 'iqr' and pt_aug == 'none':
        return os.path.join(base_path, 'best_model_dl_iqr_sampler.pkl')
    elif pt_norm == 'none' and pt_aug == 'alltrs':
        return os.path.join(base_path, 'best_model_dl_aug_sampler.pkl')
    elif pt_norm == 'none' and pt_aug == 'none':
        return os.path.join(base_path, 'best_model_dl.pkl')
    else:
        raise ValueError(f"Combinazione di pre-training non valida: pt-norm='{pt_norm}', pt-aug='{pt_aug}'")


def configure_model_for_tuning(model, tuning_strategy, num_classes):
    """
    Configura il modello per il fine-tuning.
    - Se 'lp': costruisce una testa a singolo layer e congela il resto del modello.
    - Se 'fft': costruisce una testa sequenziale e scongela l'intero modello.
    """
    logger.info(f"Configurazione del modello per la strategia: {tuning_strategy.upper()}")

    n_hidden = model.n_hidden

    if tuning_strategy == 'lp':
        # LINEAR PROBING: linear head, corpo congelato
        logger.info("Costruzione di una testa a singolo layer (nn.Sequential).")

        model.classification_head = nn.Sequential(
            nn.Linear(n_hidden, n_hidden // 2),
            nn.ReLU(),
            nn.Linear(n_hidden // 2, num_classes)
        )
        
        logger.info("Congelamento dei layer del backbone per Linear Probing.")
        # Congelamento di tutti i parametri...
        for param in model.parameters():
            param.requires_grad = False
        # Scongelamento della testa nuova di classificazione
        for param in model.classification_head.parameters():
            param.requires_grad = True

    elif tuning_strategy == 'fft':
        # FULL FINE-TUNING: testa sequential, corpo addestrabile
        logger.info("Costruzione di una testa sequenziale a due layer.")
        if n_hidden % 2 != 0:
            raise ValueError(f"n_hidden ({n_hidden}) deve essere pari per usare la testa sequenziale.")
            
        model.classification_head = nn.Sequential(
            nn.Linear(n_hidden, n_hidden // 2),
            nn.ReLU(),
            nn.Linear(n_hidden // 2, num_classes)
        )
        
        logger.info("Attivazione di tutti i layer per Full Fine-Tuning.")
        for param in model.parameters():
            param.requires_grad = True
            
    else:
        raise ValueError(f"Strategia di tuning non valida: {tuning_strategy}")
        
    # Aggiornamento del numero di classi nel modello
    model.set_n_classes(num_classes)
        
    return model



def balance_holdout_classes_npz(
    X_train_val, Y_train_val, kid_ids_train_val, action_ids_train_val,
    X_test_holdout, Y_test_holdout, kid_ids_test_holdout, action_ids_test_holdout,
    original_files_train_val, original_files_test_holdout,
    row_indices_start_train_val, row_indices_start_test_holdout,
    row_indices_end_train_val, row_indices_end_test_holdout,
    padding_counts_train_val, padding_counts_test_holdout,
    percentage_to_move=0.15, random_seed=42
):
    """
    Bilancia le classi nel test holdout spostando finestre dal train/val set.
    Lavora SOLO con etichette originali, non mappate.
    """
    logger.info("=== INIZIO BILANCIAMENTO CLASSI HOLDOUT NPZ (etichette originali) ===")

    np.random.seed(random_seed)

    train_classes = set(Y_train_val)
    test_classes = set(Y_test_holdout)
    missing_classes = train_classes - test_classes

    logger.info(f"Classi training: {sorted(train_classes)}")
    logger.info(f"Classi test: {sorted(test_classes)}")
    logger.info(f"Classi mancanti nel test: {missing_classes}")

    indices_to_move = []
    for cls in missing_classes:
        class_indices = np.where(Y_train_val == cls)[0]
        n_to_move = max(1, int(len(class_indices) * percentage_to_move))
        n_to_move = min(n_to_move, len(class_indices))  # non più di quello che c'è
        if n_to_move > 0:
            selected = np.random.choice(class_indices, n_to_move, replace=False)
            indices_to_move.extend(selected)

    if not indices_to_move:
        logger.info("Nessuna finestra da spostare.")
        return (X_train_val, Y_train_val, kid_ids_train_val, action_ids_train_val,
                original_files_train_val, row_indices_start_train_val, row_indices_end_train_val, padding_counts_train_val,
                X_test_holdout, Y_test_holdout, kid_ids_test_holdout, action_ids_test_holdout,
                original_files_test_holdout, row_indices_start_test_holdout, row_indices_end_test_holdout, padding_counts_test_holdout)

    indices_to_move = np.array(indices_to_move)

    # Estraggo dal train
    X_move = X_train_val[indices_to_move]
    Y_move = Y_train_val[indices_to_move]
    kid_move = kid_ids_train_val[indices_to_move]
    act_move = action_ids_train_val[indices_to_move]
    files_move = original_files_train_val[indices_to_move]
    row_start_move = row_indices_start_train_val[indices_to_move]
    row_end_move = row_indices_end_train_val[indices_to_move]
    pad_move = padding_counts_train_val[indices_to_move]

    # Train aggiornato
    mask = np.ones(len(X_train_val), dtype=bool)
    mask[indices_to_move] = False
    X_train_val = X_train_val[mask]
    Y_train_val = Y_train_val[mask]
    kid_ids_train_val = kid_ids_train_val[mask]
    action_ids_train_val = action_ids_train_val[mask]
    original_files_train_val = original_files_train_val[mask]
    row_indices_start_train_val = row_indices_start_train_val[mask]
    row_indices_end_train_val = row_indices_end_train_val[mask]
    padding_counts_train_val = padding_counts_train_val[mask]

    # Test aggiornato
    X_test_holdout = np.concatenate([X_test_holdout, X_move])
    Y_test_holdout = np.concatenate([Y_test_holdout, Y_move])
    kid_ids_test_holdout = np.concatenate([kid_ids_test_holdout, kid_move])
    action_ids_test_holdout = np.concatenate([action_ids_test_holdout, act_move])
    original_files_test_holdout = np.concatenate([original_files_test_holdout, files_move])
    row_indices_start_test_holdout = np.concatenate([row_indices_start_test_holdout, row_start_move])
    row_indices_end_test_holdout = np.concatenate([row_indices_end_test_holdout, row_end_move])
    padding_counts_test_holdout = np.concatenate([padding_counts_test_holdout, pad_move])

    logger.info("=== DISTRIBUZIONE FINALE ===")
    logger.info(f"Train classi: {Counter(Y_train_val)}")
    logger.info(f"Test classi: {Counter(Y_test_holdout)}")

    # 🔒 Controllo forte
    missing_after = train_classes - set(Y_test_holdout)
    if missing_after:
        logger.error(f"Ancora classi mancanti nel test: {missing_after}")
        raise RuntimeError("Bilanciamento non riuscito!")

    logger.info("✓ Tutte le classi ora presenti nel test set.")
    return (X_train_val, Y_train_val, kid_ids_train_val, action_ids_train_val,
            original_files_train_val, row_indices_start_train_val, row_indices_end_train_val, padding_counts_train_val,
            X_test_holdout, Y_test_holdout, kid_ids_test_holdout, action_ids_test_holdout,
            original_files_test_holdout, row_indices_start_test_holdout, row_indices_end_test_holdout, padding_counts_test_holdout)
