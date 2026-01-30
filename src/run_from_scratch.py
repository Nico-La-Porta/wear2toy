import os
import argparse
import pickle
import random
from collections import Counter
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

import optuna
from optuna.pruners import MedianPruner

from src.app_config import DOWNSTREAM_DATA_DIR, REPORTS_DIR, FIGURES_DIR, MODELS_DIR
from src.models.DeepConvLSTM import DeepConvLSTM, HARDataset, collate_fn
from src.utils import mapping_activity, normalization, train_with_cm
from src.utils.log_config import setup_logging, logger
from src.utils.focal_loss import FocalLoss, LabelSmoothingCrossEntropy, CombinedLoss
from src.utils.figures import combine_kfold_confusion_matrices
from src.utils.transformations import *
from src.utils.transformations_utils import *
from src.utils.data_loader import load_and_preprocess_data_unified, add_consecutive_segment_id


def create_single_distribution_bar_chart(Y, toy_name="Dataset", save_path=None, title_suffix=" ", use_class_prefix=True):
    distribution = Counter(Y)
    all_classes = sorted(distribution.keys())
    total_samples = len(Y)
    counts = [distribution[cls] for cls in all_classes]
    percentages = [(count / total_samples) * 100 for count in counts]
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#2C3E50', '#E74C3C', '#3498DB', '#27AE60', '#F39C12', '#9B59B6', '#1ABC9C', '#34495E']
    bar_colors = [colors[i % len(colors)] for i in range(len(all_classes))]
    bars = ax.bar(range(len(all_classes)), percentages, color=bar_colors, alpha=0.8, edgecolor='black', linewidth=0.8, hatch='///')
    ax.set_xlabel('Activity Class', fontsize=16, fontweight='bold')
    ax.set_ylabel('Percentage (%)', fontsize=16, fontweight='bold')
    ax.set_title(f'{toy_name}  {title_suffix}', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(range(len(all_classes)))
    if use_class_prefix and all(isinstance(cls, (int, float)) for cls in all_classes):
        ax.set_xticklabels([f'C{cls}' for cls in all_classes], fontsize=12)
    else:
        ax.set_xticklabels(all_classes, fontsize=12, rotation=45, ha='right')
    ax.tick_params(axis='y', labelsize=12)
    ax.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    for i, (bar, percentage, count) in enumerate(zip(bars, percentages, counts)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
               f'{percentage:.1f}%\n({count})', ha='center', va='bottom',
               fontsize=10, fontweight='bold')
    plt.tight_layout()
    if save_path:
        base_path = save_path.rsplit('.', 1)[0] if '.' in save_path else save_path
        plt.savefig(f"{base_path}.pdf", dpi=300, bbox_inches='tight', format='pdf', facecolor='white', edgecolor='none')
        plt.savefig(f"{base_path}.png", dpi=300, bbox_inches='tight', format='png', facecolor='white', edgecolor='none')
        plt.savefig(f"{base_path}.svg", bbox_inches='tight', format='svg', facecolor='white', edgecolor='none')
    return fig

def log_and_plot_distribution(Y, title_prefix, toy_name, toy_mapping, save_dir):
    logger.info(f"===== ANALISI DISTRIBUZIONE: {title_prefix} - {toy_name.upper()} =====")
    if len(Y) == 0:
        logger.warning("L'array Y è vuoto. Impossibile mostrare la distribuzione.")
        return
    try:
        logger.info("Conversione ID -> nomi tramite 'convert_original_to_names'...")
        Y_names = mapping_activity.convert_original_to_names(Y, toy_name.upper())
    except Exception as e:
        logger.error(f"Errore conversione nomi: {e}")
        labels_dict = toy_mapping["encoded_to_name"]
        Y_names = [labels_dict.get(int(y), f"ID_{y}") for y in Y]
    file_suffix = title_prefix.lower().replace(" ", "_").replace("(", "").replace(")", "")
    save_path_base = os.path.join(save_dir, f"{toy_name}_distribuzione_{file_suffix}")
    fig = create_single_distribution_bar_chart(
        Y=Y_names, toy_name=toy_name.upper(),
        title_suffix=f"({title_prefix})",
        save_path=save_path_base, use_class_prefix=False
    )
    if fig:
        plt.close(fig)


def balance_holdout_classes_npz(X_train_val, Y_train_val, kid_ids_train_val, action_ids_train_val,
                               X_test_holdout, Y_test_holdout, kid_ids_test_holdout, action_ids_test_holdout,
                               original_files_train_val, original_files_test_holdout,
                               row_indices_start_train_val, row_indices_start_test_holdout,
                               row_indices_end_train_val, row_indices_end_test_holdout,
                               padding_counts_train_val, padding_counts_test_holdout,
                               percentage_to_move=0.15, random_seed=42):
    """
    Bilancia le classi nel test holdout spostando finestre dal train/val set
    quando alcune classi sono mancanti nel test set (workflow NPZ).
    
    Args:
        X_train_val, Y_train_val: Dati di train/validation mappati
        kid_ids_train_val, action_ids_train_val: Metadati train/val
        X_test_holdout, Y_test_holdout: Dati di test holdout mappati
        kid_ids_test_holdout, action_ids_test_holdout: Metadati test holdout
        original_files_*, row_indices_*, padding_counts_*: Altri metadati NPZ
        percentage_to_move: Percentuale di finestre da spostare per classe mancante
        random_seed: Seed per riproducibilità
    
    Returns:
        tuple: Dati e metadati aggiornati per train_val e test_holdout
    """
    logger.info("=== INIZIO BILANCIAMENTO CLASSI HOLDOUT NPZ ===")
    
    # Conta distribuzione classi
    train_class_distribution = Counter(Y_train_val)
    test_class_distribution = Counter(Y_test_holdout)
    
    # Trova classi mancanti nel test
    all_classes = set(Y_train_val)
    missing_classes = [cls for cls in all_classes if cls not in test_class_distribution]
    
    logger.info(f"Classi totali nel training: {sorted(all_classes)}")
    logger.info(f"Classi presenti nel test: {sorted(test_class_distribution.keys())}")
    logger.info(f"Classi mancanti nel test: {missing_classes}")
    
    if not missing_classes:
        logger.info("Tutte le classi sono già presenti nel test. Nessun bilanciamento necessario.")
        return (X_train_val, Y_train_val, kid_ids_train_val, action_ids_train_val,
                original_files_train_val, row_indices_start_train_val, row_indices_end_train_val, 
                padding_counts_train_val,
                X_test_holdout, Y_test_holdout, kid_ids_test_holdout, action_ids_test_holdout,
                original_files_test_holdout, row_indices_start_test_holdout, row_indices_end_test_holdout,
                padding_counts_test_holdout)
    
    # Calcola finestre da spostare per ogni classe mancante
    windows_to_move = {}
    for missing_class in missing_classes:
        total_windows = train_class_distribution[missing_class]
        windows_to_move[missing_class] = max(1, int(total_windows * percentage_to_move))
        logger.info(f"Classe {missing_class}: {total_windows} finestre totali -> "
                   f"{windows_to_move[missing_class]} finestre da spostare")
    
    # Seleziona indici da spostare
    np.random.seed(random_seed)
    indices_to_move = []
    
    for missing_class in missing_classes:
        # Trova tutti gli indici delle finestre di questa classe nel training
        class_indices = np.where(Y_train_val == missing_class)[0]
        
        # Seleziona casualmente le finestre da spostare
        n_to_move = windows_to_move[missing_class]
        if n_to_move > 0 and len(class_indices) >= n_to_move:
            selected_indices = np.random.choice(class_indices, size=n_to_move, replace=False)
            indices_to_move.extend(selected_indices)
            logger.info(f"Selezionate {len(selected_indices)} finestre per classe {missing_class}")
        elif len(class_indices) < n_to_move:
            logger.warning(f"Classe {missing_class}: richieste {n_to_move} finestre ma disponibili solo {len(class_indices)}")
            indices_to_move.extend(class_indices)  # Sposta tutte le finestre disponibili
    
    indices_to_move = np.array(indices_to_move, dtype=int)
    logger.info(f"Totale finestre da spostare: {len(indices_to_move)}")
    
    if len(indices_to_move) == 0:
        logger.info("Nessuna finestra da spostare.")
        return (X_train_val, Y_train_val, kid_ids_train_val, action_ids_train_val,
                original_files_train_val, row_indices_start_train_val, row_indices_end_train_val, 
                padding_counts_train_val,
                X_test_holdout, Y_test_holdout, kid_ids_test_holdout, action_ids_test_holdout,
                original_files_test_holdout, row_indices_start_test_holdout, row_indices_end_test_holdout,
                padding_counts_test_holdout)
    
    # ===== ESTRAI FINESTRE DA SPOSTARE =====
    X_to_move = X_train_val[indices_to_move]
    Y_to_move = Y_train_val[indices_to_move]
    kid_ids_to_move = kid_ids_train_val[indices_to_move]
    action_ids_to_move = action_ids_train_val[indices_to_move]
    original_files_to_move = original_files_train_val[indices_to_move]
    row_start_to_move = row_indices_start_train_val[indices_to_move]
    row_end_to_move = row_indices_end_train_val[indices_to_move]
    padding_to_move = padding_counts_train_val[indices_to_move]
    
    # ===== CREA MASCHERA PER FINESTRE DA MANTENERE NEL TRAINING =====
    mask_keep = np.ones(len(X_train_val), dtype=bool)
    mask_keep[indices_to_move] = False
    
    # ===== AGGIORNA TRAINING SET (rimuove finestre spostate) =====
    X_train_updated = X_train_val[mask_keep]
    Y_train_updated = Y_train_val[mask_keep]
    kid_ids_train_updated = kid_ids_train_val[mask_keep]
    action_ids_train_updated = action_ids_train_val[mask_keep]
    original_files_train_updated = original_files_train_val[mask_keep]
    row_start_train_updated = row_indices_start_train_val[mask_keep]
    row_end_train_updated = row_indices_end_train_val[mask_keep]
    padding_train_updated = padding_counts_train_val[mask_keep]
    
    # ===== AGGIORNA TEST SET (aggiunge finestre spostate) =====
    X_test_updated = np.concatenate([X_test_holdout, X_to_move], axis=0)
    Y_test_updated = np.concatenate([Y_test_holdout, Y_to_move], axis=0)
    kid_ids_test_updated = np.concatenate([kid_ids_test_holdout, kid_ids_to_move], axis=0)
    action_ids_test_updated = np.concatenate([action_ids_test_holdout, action_ids_to_move], axis=0)
    original_files_test_updated = np.concatenate([original_files_test_holdout, original_files_to_move], axis=0)
    row_start_test_updated = np.concatenate([row_indices_start_test_holdout, row_start_to_move], axis=0)
    row_end_test_updated = np.concatenate([row_indices_end_test_holdout, row_end_to_move], axis=0)
    padding_test_updated = np.concatenate([padding_counts_test_holdout, padding_to_move], axis=0)
    
    # ===== LOG RISULTATI FINALI =====
    logger.info("=== RISULTATI BILANCIAMENTO NPZ ===")
    logger.info(f"Training set: {X_train_val.shape} -> {X_train_updated.shape}")
    logger.info(f"Test set: {X_test_holdout.shape} -> {X_test_updated.shape}")
    
    # Verifica nuova distribuzione
    updated_train_distribution = Counter(Y_train_updated)
    updated_test_distribution = Counter(Y_test_updated)
    
    logger.info("Nuova distribuzione training:")
    for class_label in sorted(updated_train_distribution.keys()):
        logger.info(f"   Classe {class_label}: {updated_train_distribution[class_label]} finestre")
    
    logger.info("Nuova distribuzione test:")
    for class_label in sorted(updated_test_distribution.keys()):
        logger.info(f"   Classe {class_label}: {updated_test_distribution[class_label]} finestre")
    
    # Verifica finale
    still_missing = [cls for cls in all_classes if cls not in updated_test_distribution]
    if still_missing:
        logger.warning(f"Classi ancora mancanti nel test: {still_missing}")
    else:
        logger.info("✓ Tutte le classi ora presenti nel test set.")
    
    # ===== DEBUG COERENZA METADATI =====
    logger.info("=== VERIFICA COERENZA METADATI ===")
    logger.info(f"X_train shape: {X_train_updated.shape}")
    logger.info(f"kid_ids_train length: {len(kid_ids_train_updated)}")
    logger.info(f"action_ids_train length: {len(action_ids_train_updated)}")
    logger.info(f"X_test shape: {X_test_updated.shape}")
    logger.info(f"kid_ids_test length: {len(kid_ids_test_updated)}")
    logger.info(f"action_ids_test length: {len(action_ids_test_updated)}")
    
    # Verifica che tutte le lunghezze siano coerenti
    assert len(X_train_updated) == len(Y_train_updated) == len(kid_ids_train_updated), "Lunghezze training inconsistenti"
    assert len(X_test_updated) == len(Y_test_updated) == len(kid_ids_test_updated), "Lunghezze test inconsistenti"
    
    logger.info("✓ Verifica coerenza metadati: OK")
    
    return (X_train_updated, Y_train_updated, kid_ids_train_updated, action_ids_train_updated,
            original_files_train_updated, row_start_train_updated, row_end_train_updated, 
            padding_train_updated,
            X_test_updated, Y_test_updated, kid_ids_test_updated, action_ids_test_updated,
            original_files_test_updated, row_start_test_updated, row_end_test_updated,
            padding_test_updated)


# ===== Configura il modello per LP / FFT =====
def configure_model_for_tuning(model, tuning_strategy, num_classes):
    logger.info(f"Configurazione del modello per la strategia: {tuning_strategy.upper()}")
    n_hidden = model.n_hidden
    if tuning_strategy == 'lp':
        model.classification_head = nn.Sequential(
            nn.Linear(n_hidden, n_hidden // 2),
            nn.ReLU(),
            nn.Linear(n_hidden // 2, num_classes)
        )
        for p in model.parameters():
            p.requires_grad = False
        for p in model.classification_head.parameters():
            p.requires_grad = True
    elif tuning_strategy == 'fft':
        if n_hidden % 2 != 0:
            raise ValueError(f"n_hidden ({n_hidden}) deve essere pari per usare la testa sequenziale.")
        model.classification_head = nn.Sequential(
            nn.Linear(n_hidden, n_hidden // 2),
            nn.ReLU(),
            nn.Linear(n_hidden // 2, num_classes)
        )
        for p in model.parameters():
            p.requires_grad = True
    else:
        raise ValueError(f"Strategia di tuning non valida: {tuning_strategy}")
    model.set_n_classes(num_classes)
    return model

# ====== Pipeline di un singolo esperimento  ======
def run_single_experiment(args, seed_used, target_actions, ft_norm, ft_aug, tuning_strategy):
    # ========== Setup nome esperimento e cartelle ==========
    # ========== Setup nome esperimento e cartelle ==========
    exp_name = f"FS_T{args.toy}_FTN{ft_norm}_FTA{ft_aug}_TS{tuning_strategy}"
    project_root = os.path.dirname(os.path.abspath(__file__))
    logs_exp_dir = os.path.join(project_root, "logs", exp_name)
    os.makedirs(logs_exp_dir, exist_ok=True)
    
    # Includi seed nel setup del logging
    seed_tag = f"seed_{seed_used}"
    setup_logging(f"{exp_name}/{seed_tag}")
    
    exp_root_models  = os.path.join(MODELS_DIR,  exp_name, seed_tag)
    exp_root_figures = os.path.join(FIGURES_DIR, exp_name, seed_tag)
    exp_root_reports = os.path.join(REPORTS_DIR, exp_name, seed_tag)
    for p in [exp_root_models, exp_root_figures, exp_root_reports]:
        os.makedirs(p, exist_ok=True)


    logger.info(f"===== INIZIO ESPERIMENTO (FROM-SCRATCH): {exp_name} =====")
    logger.info(f"Seed: {seed_used}")
    logger.info(f"Cartelle: {exp_root_models}, {exp_root_figures}, {exp_root_reports}")

    # ===== CARICA DATI ORIGINALI PER CALCOLO STATISTICHE =====
    data_path = DOWNSTREAM_DATA_DIR
    logger.info("Caricamento dati originali per calcolo statistiche di normalizzazione...")
    df_toy, toy_mapping = load_and_preprocess_data_unified(
        toy_name=args.toy, 
        data_path=data_path,
        target_actions=target_actions 
    )

    df_toy = add_consecutive_segment_id(df_toy)
    logger.info(f"Dati originali caricati: {df_toy.shape}")
    logger.info(f"Classi presenti: {sorted(df_toy['action_id'].unique())}")

    # ===== CARICA FINESTRE PRE-PROCESSATE DA NPZ =====
    pkl_dir = os.path.dirname(args.filtered_indices_pkl)
    npz_filename = f'{args.toy}_processed_windows_seed{seed_used}.npz'
    npz_path = os.path.join(pkl_dir, npz_filename)
    
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"File NPZ non trovato: {npz_path}. Assicurati di aver eseguito zupt_analysis_2.py prima.")

    logger.info(f"Caricamento finestre pre-processate da: {npz_path}")
    try:
        data = np.load(npz_path, allow_pickle=True)
        
        # Dati principali
        X = data['X']
        Y = data['Y'] 
        kid_ids = data['kid_ids']
        action_ids = data['action_ids']
        original_files = data['original_files']
        
        # Metadati aggiuntivi
        row_indices_start = data['row_indices_start']
        row_indices_end = data['row_indices_end'] 
        padding_counts = data['padding_counts']
        
        # Verifica coerenza
        assert len(X) == len(Y) == len(kid_ids) == len(action_ids), "Lunghezze inconsistenti nel NPZ"
        
        logger.info(f"Finestre caricate con successo:")
        logger.info(f"  - Shape finestre: {X.shape}")
        logger.info(f"  - Numero etichette: {len(Y)}")
        logger.info(f"  - Bambini unici: {sorted(np.unique(kid_ids))}")
        logger.info(f"  - Azioni uniche: {sorted(np.unique(action_ids))}")
        logger.info(f"  - Finestre con padding: {np.sum(padding_counts > 0)}")
        
    except Exception as e:
        raise RuntimeError(f"Errore nel caricamento del file NPZ: {e}")
    

    # ========== CALCOLO STATISTICHE DI NORMALIZZAZIONE ==========
    mean, std = None, None
    
    if ft_norm == 'meanstdonft':
        logger.info("Calcolo statistiche di normalizzazione sul dataset di fine-tuning...")
        
        # Filtra solo i bambini di training per calcolare le statistiche
        if args.holdout_kids:
            df_train_for_stats = df_toy[~df_toy['kid_id'].isin(args.holdout_kids)]
            logger.info(f"Calcolo statistiche solo sui bambini di training (escludendo holdout: {args.holdout_kids})")
        else:
            df_train_for_stats = df_toy
            logger.info("Calcolo statistiche su tutto il dataset")
            
        mean, std = normalization.compute_dataframe_mean_std(df_train_for_stats)
        logger.info(f"Statistiche calcolate - Mean: {mean[:3]}..., Std: {std[:3]}...")
    elif ft_norm == 'none':
        logger.info("Nessuna normalizzazione applicata al fine-tuning")
    else:
        raise ValueError(f"ft_norm non supportata in from-scratch: {ft_norm}")
    

    # ===== APPLICAZIONE NORMALIZZAZIONE ALLE FINESTRE =====
    if mean is not None and std is not None:
        logger.info("Applicazione normalizzazione alle finestre pre-processate...")
        
        # Normalizza tutte le finestre
        original_shape = X.shape
        X_reshaped = X.reshape(-1, X.shape[-1])  # (n_windows * timesteps, n_features)
        
        # Applica normalizzazione
        X_normalized = (X_reshaped - mean) / std
        X = X_normalized.reshape(original_shape)  # Ripristina forma originale
        
        logger.info(f"Normalizzazione applicata a tutte le finestre")
        logger.info(f"Shape dopo normalizzazione: {X.shape}")
    else:
        logger.info("Nessuna normalizzazione applicata alle finestre")
    
    # ========== SPLIT HOLDOUT VS KFOLD ==========
    if args.holdout_kids:
        logger.info(f"Separazione bambini per holdout test: {args.holdout_kids}")

        # Crea maschere per dividere train/val e test holdout
        holdout_mask = np.isin(kid_ids, args.holdout_kids)
        train_val_mask = ~holdout_mask
        
        # ===== SPLIT INIZIALE =====
        # Dati training/validation
        X_train_val = X[train_val_mask]
        Y_train_val = Y[train_val_mask]
        kid_ids_train_val = kid_ids[train_val_mask]
        action_ids_train_val = action_ids[train_val_mask]
        original_files_train_val = original_files[train_val_mask]
        row_start_train_val = row_indices_start[train_val_mask]
        row_end_train_val = row_indices_end[train_val_mask]
        padding_train_val = padding_counts[train_val_mask]
        
        # Dati holdout test
        X_test_holdout = X[holdout_mask]
        Y_test_holdout = Y[holdout_mask]
        kid_ids_test_holdout = kid_ids[holdout_mask]
        action_ids_test_holdout = action_ids[holdout_mask]
        original_files_test_holdout = original_files[holdout_mask]
        row_start_test_holdout = row_indices_start[holdout_mask]
        row_end_test_holdout = row_indices_end[holdout_mask]
        padding_test_holdout = padding_counts[holdout_mask]
        
        logger.info(f"Training/Validation set iniziale: {X_train_val.shape}")
        logger.info(f"Holdout Test set iniziale: {X_test_holdout.shape}")
        logger.info(f"Bambini in training: {sorted(np.unique(kid_ids_train_val))}")
        logger.info(f"Bambini in test: {sorted(np.unique(kid_ids_test_holdout))}")

        # ===== MAPPATURA AZIONI =====
        logger.info("Mappatura delle azioni per la classificazione...")
        
        # Usa solo le azioni del training per creare la mappatura
        unique_labels = np.unique(Y_train_val)
        label_mapping = {label: i for i, label in enumerate(unique_labels)}
        Y_train_val_mapped = np.array([label_mapping[y] for y in Y_train_val])

        # Mappa anche le etichette del test holdout
        Y_test_holdout_mapped = np.array([label_mapping[y] for y in Y_test_holdout])
        num_classes = len(unique_labels)
        
        logger.info(f"Azioni originali: {sorted(unique_labels)}")
        logger.info(f"Mappatura azioni: {label_mapping}")
        logger.info(f"Numero classi finali: {num_classes}")
        
        # ===== VERIFICA NECESSITÀ BILANCIAMENTO =====
        train_actions_set = set(Y_train_val_mapped)
        test_actions_set = set(Y_test_holdout_mapped)
        missing_in_test = train_actions_set - test_actions_set

        if missing_in_test:
            logger.info(f"Applicazione bilanciamento per classi mancanti nel test: {missing_in_test}")

            # ===== APPLICA BILANCIAMENTO NPZ =====
            (X_train_val, Y_train_val_mapped, kid_ids_train_val, action_ids_train_val,
             original_files_train_val, row_start_train_val, row_end_train_val, padding_train_val,
             X_test_holdout, Y_test_holdout_mapped, kid_ids_test_holdout, action_ids_test_holdout,
             original_files_test_holdout, row_start_test_holdout, row_end_test_holdout, 
             padding_test_holdout) = balance_holdout_classes_npz(
                X_train_val, Y_train_val_mapped, kid_ids_train_val, action_ids_train_val,
                X_test_holdout, Y_test_holdout_mapped, kid_ids_test_holdout, action_ids_test_holdout,
                original_files_train_val, original_files_test_holdout,
                row_start_train_val, row_start_test_holdout,
                row_end_train_val, row_end_test_holdout,
                padding_train_val, padding_test_holdout,
                percentage_to_move=0.15, random_seed=seed_used
            )
            
            logger.info("Bilanciamento applicato con successo!")
        else:
            # Nessun bilanciamento necessario, ma dobbiamo comunque usare le etichette mappate
            Y_train_val_mapped = Y_train_val_mapped  # già calcolato sopra
            logger.info("Nessun bilanciamento necessario: tutte le classi presenti nel test")
        

        # ===== RICOSTRUISCI METADATI PER COMPATIBILITÀ =====
        # Crea metadati nel formato atteso dal resto del codice
        all_window_indices = []
        all_consecutivity = np.ones(len(X_train_val), dtype=bool)  # Assume tutte consecutive
        
        for i in range(len(X_train_val)):
            all_window_indices.append({
                'global_window_id': i,
                'action_id': int(action_ids_train_val[i]),
                'kid_id': str(kid_ids_train_val[i]),
                'original_file': str(original_files_train_val[i]),
                'row_indices': list(range(int(row_start_train_val[i]), int(row_end_train_val[i]) + 1)) if row_start_train_val[i] != -1 else [],
                'padding_count': int(padding_train_val[i])
            })
        
        # Aggiorna le variabili principali per il resto del workflow
        X = X_train_val
        Y_mapped = Y_train_val_mapped
        logger.info(f"Dati finali per K-Fold: X.shape={X.shape}, Y_mapped.shape={Y_mapped.shape}")
        logger.info(f"Dati holdout per test finale: X_test.shape={X_test_holdout.shape}")
        
        # Metadati test holdout per valutazione finale
        all_window_indices_test = []
        all_consecutivity_test = np.ones(len(X_test_holdout), dtype=bool)
        
        for i in range(len(X_test_holdout)):
            all_window_indices_test.append({
                'global_window_id': i,
                'action_id': int(action_ids_test_holdout[i]),
                'kid_id': str(kid_ids_test_holdout[i]),
                'original_file': str(original_files_test_holdout[i]),
                'row_indices': list(range(int(row_start_test_holdout[i]), int(row_end_test_holdout[i]) + 1)) if row_start_test_holdout[i] != -1 else [],
                'padding_count': int(padding_test_holdout[i])
            })
    else:
        logger.info("Nessun holdout specificato: usando tutto il dataset per K-Fold")
        X_test_holdout = None
        Y_test_holdout_mapped = None

        # ===== MAPPATURA AZIONI (caso senza holdout) =====
        unique_labels = np.unique(Y)
        label_mapping = {label: i for i, label in enumerate(unique_labels)}
        Y_mapped = np.array([label_mapping[y] for y in Y])
        num_classes = len(unique_labels)
        
        # Ricostruisci metadati standard
        all_window_indices = []
        all_consecutivity = np.ones(len(X), dtype=bool)
        
        for i in range(len(X)):
            all_window_indices.append({
                'global_window_id': i,
                'action_id': int(action_ids[i]),
                'kid_id': str(kid_ids[i]),
                'original_file': str(original_files[i]),
                'row_indices': list(range(int(row_indices_start[i]), int(row_indices_end[i]) + 1)) if row_indices_start[i] != -1 else [],
                'padding_count': int(padding_counts[i])
            })
    
     # Crea dizionario per visualizzazione 
    toy_mapping_orig = mapping_activity.get_toy_mapping(args.toy.upper())
    labels_dict_mapped = {}
    
    for mapped_id in range(num_classes):
        # Trova l'ID originale corrispondente
        original_id = None
        for orig, mapped in label_mapping.items():
            if mapped == mapped_id:
                original_id = orig
                break
        
        # Ottieni il nome della classe dall'ID originale
        if original_id is not None:
            class_name = toy_mapping_orig["original_to_name"].get(int(original_id), f"Unknown_{original_id}")
            labels_dict_mapped[mapped_id] = class_name
        else:
            labels_dict_mapped[mapped_id] = f"Class_{mapped_id}"
    
    logger.info(f"Dizionario classi per visualizzazione: {labels_dict_mapped}")

    # ========== Remapping etichette e class weights ==========
    class_weights = compute_class_weight('balanced', classes=np.unique(Y_mapped), y=Y_mapped)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
    logger.info(f"Pesi classi: {dict(zip(np.unique(Y_mapped), class_weights))}")

    # ===== DISTRIBUZIONE INIZIALE =====
    # Converti Y_mapped di nuovo alle azioni originali per la visualizzazione
    inv_label_mapping = {v: k for k, v in label_mapping.items()}
    Y_original_for_plot = np.array([inv_label_mapping[y] for y in Y_mapped])
    
    log_and_plot_distribution(
        Y=Y_original_for_plot,
        title_prefix="Distribuzione Finestre da NPZ",
        toy_name=args.toy,
        toy_mapping=toy_mapping,
        save_dir=exp_root_figures
    )

    # ========== Hold-out branch ==========
    if args.holdout_kids:
        # ====== KFold per HP, test interno per fold e training finale su tutto ======
        skf = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=seed_used)
        all_fold_scores = []
        fold_results = {'fold': [], 'best_f1_score': [], 'best_params': [], 'test_f1_score': []}

        for fold, (train_val_idx, test_idx) in enumerate(skf.split(X, Y_mapped)):
            X_train_val_fold, X_test_fold = X[train_val_idx], X[test_idx]
            Y_train_val_fold, Y_test_fold = Y_mapped[train_val_idx], Y_mapped[test_idx]
            train_val_indices_meta = [all_window_indices[i] for i in train_val_idx]
            test_indices_meta = [all_window_indices[i] for i in test_idx]
            train_val_consec = all_consecutivity[train_val_idx]
            test_consec = all_consecutivity[test_idx]

            split_idx = int(0.8 * len(X_train_val_fold))
            X_train_fold, Y_train_fold = X_train_val_fold[:split_idx], Y_train_val_fold[:split_idx]
            X_val_fold,   Y_val_fold   = X_train_val_fold[split_idx:], Y_train_val_fold[split_idx:]
            train_indices_meta = train_val_indices_meta[:split_idx]
            val_indices_meta   = train_val_indices_meta[split_idx:]
            train_consec       = train_val_consec[:split_idx]
            val_consec         = train_val_consec[split_idx:]

            if ft_aug == 'alltrs':
                logger.info("Applico data augmentation al set di training (fold).")
                transformation_function = generate_composite_transform_function_simple([
                    noise_transform_vectorized, scaling_transform_vectorized, negate_transform_vectorized,
                    time_flip_transform_vectorized, intra_sensor_channel_shuffle_transform_vectorized,
                    time_warp_transform_improved, time_warp_transform_low_cost
                ])
                X_train_aug = transformation_function(X_train_fold)
                X_train_fold = np.concatenate([X_train_fold, X_train_aug], axis=0)
                Y_train_fold = np.concatenate([Y_train_fold, Y_train_fold.copy()], axis=0)
                train_indices_meta.extend(train_indices_meta.copy())
                train_consec = np.concatenate([train_consec, train_consec.copy()], axis=0)

            train_dataset = HARDataset(X_train_fold, Y_train_fold, train_indices_meta, train_consec)
            val_dataset   = HARDataset(X_val_fold,   Y_val_fold,   val_indices_meta,   val_consec)
            test_dataset  = HARDataset(X_test_fold,  Y_test_fold,  test_indices_meta,  test_consec)

            def objective_fold(trial):
                lr = trial.suggest_categorical('lr', [5e-4, 1e-3, 5e-3, 1e-2])
                batch_size = trial.suggest_categorical('batch_size', [2, 4])
                loss_type = trial.suggest_categorical('loss_type', ['focal', 'label_smoothing', 'weighted_ce', 'combined'])
                if loss_type == 'focal':
                    gamma = trial.suggest_categorical('gamma', [1.0, 2.0, 3.0])
                    criterion = FocalLoss(gamma=gamma, num_classes=num_classes, task_type='multi-class')
                elif loss_type == 'label_smoothing':
                    smoothing = trial.suggest_categorical('smoothing', [0.05, 0.1, 0.15, 0.2])
                    criterion = LabelSmoothingCrossEntropy(epsilon=smoothing)
                elif loss_type == 'combined':
                    gamma = trial.suggest_categorical('gamma_combined', [1.0, 2.0])
                    focal_weight = trial.suggest_categorical('focal_weight', [0.3, 0.5, 0.7])
                    criterion = CombinedLoss(focal_weight=focal_weight, ce_weight=1.0 - focal_weight,
                                             gamma=gamma, class_weights=class_weights_tensor, num_classes=num_classes)
                else:
                    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, collate_fn=collate_fn)
                val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=collate_fn)
                model = DeepConvLSTM()                            # <= from scratch (NO pesi pretrain)
                model = configure_model_for_tuning(model, tuning_strategy, num_classes)
                best_f1, best_state = train_with_cm.train(
                    net=model, train_loader=train_loader, val_loader=val_loader,
                    exp_figures_dir=exp_root_figures, exp_reports_dir=exp_root_reports,
                    epochs=args.epochs, lr=lr, criterion=criterion, save_plots=False
                )
                trial.set_user_attr("best_state", best_state)
                return best_f1

            study = optuna.create_study(direction='maximize', pruner=MedianPruner())
            study.optimize(objective_fold, n_trials=args.n_trials)
            best_params_fold = study.best_params
            best_f1_fold = study.best_value
            best_state = study.best_trial.user_attrs["best_state"]
            logger.info(f"[Fold {fold+1}] Best params: {best_params_fold} | Val F1: {best_f1_fold:.4f}")

            model_final_fold = DeepConvLSTM()
            model_final_fold = configure_model_for_tuning(model_final_fold, tuning_strategy, num_classes)
            model_final_fold.load_state_dict(best_state, strict=False)
            test_loader = DataLoader(test_dataset, batch_size=best_params_fold['batch_size'], shuffle=False, drop_last=False, collate_fn=collate_fn)
            _, _, test_f1_fold = train_with_cm.evaluate_model(
                model_final_fold, test_loader,
                figure_name=f"cm_test_fold_{fold + 1}_{exp_name}",
                save_confusion_matrix=True, save_predictions_csv=True, save_f1_score=True,
                labels_dict=labels_dict_mapped, exp_figures_dir=exp_root_figures, exp_reports_dir=exp_root_reports
            )
            model_path_fold = os.path.join(exp_root_models, f"best_model_fold_{fold+1}.pth")
            torch.save(model_final_fold.state_dict(), model_path_fold)
            fold_results['fold'].append(fold + 1)
            fold_results['best_f1_score'].append(best_f1_fold)
            fold_results['best_params'].append(best_params_fold)
            fold_results['test_f1_score'].append(test_f1_fold)
            all_fold_scores.append({'fold': fold + 1, 'params': best_params_fold, 'val_f1': best_f1_fold,
                                    'test_f1': test_f1_fold, 'model_path': model_path_fold})

        # Scelgo i migliori HP e addestro su tutto Train/Val
        best_fold = max(all_fold_scores, key=lambda x: x['val_f1'])
        best_hyperparameters = best_fold['params']
        logger.info(f"Migliori iperparametri (da Fold {best_fold['fold']}): {best_hyperparameters}")
        full_train_val_dataset = HARDataset(X, Y_mapped, all_window_indices, all_consecutivity)
        final_train_loader = DataLoader(full_train_val_dataset, batch_size=best_hyperparameters['batch_size'],
                                        shuffle=True, collate_fn=collate_fn)
        final_model = DeepConvLSTM()
        final_model = configure_model_for_tuning(final_model, tuning_strategy, num_classes)
        final_f1_train, final_best_state = train_with_cm.train(
            net=final_model, train_loader=final_train_loader, val_loader=None,
            exp_figures_dir=exp_root_figures, exp_reports_dir=exp_root_reports,
            epochs=args.epochs, lr=best_hyperparameters['lr'],
            figure_name="final_training_on_all_train_val_data", save_plots=False
        )
        final_model.load_state_dict(final_best_state, strict=False)
        final_model_path = os.path.join(exp_root_models, "final_model.pkl")
        torch.save(final_model.state_dict(), final_model_path)
        logger.info(f"Modello finale salvato in: {final_model_path}")
        final_eval_loader = DataLoader(full_train_val_dataset, batch_size=best_hyperparameters['batch_size'],
                                       shuffle=False, collate_fn=collate_fn)
        train_with_cm.evaluate_model(
            net=final_model, test_loader=final_eval_loader,
            exp_figures_dir=exp_root_figures, exp_reports_dir=exp_root_reports,
            figure_name="FINAL_training_evaluation",
            save_confusion_matrix=True, save_predictions_csv=True, save_f1_score=True,
            labels_dict=labels_dict_mapped
        )
        test_dataset_final = HARDataset(X_test_holdout, Y_test_holdout_mapped, window_indices=all_window_indices_test,
                                        consecutivity=all_consecutivity_test)
        test_loader_final = DataLoader(test_dataset_final, batch_size=best_hyperparameters['batch_size'],
                                       collate_fn=collate_fn)
        train_with_cm.evaluate_model(
            net=final_model, test_loader=test_loader_final,
            exp_figures_dir=exp_root_figures, exp_reports_dir=exp_root_reports,
            figure_name="FINAL_holdout_evaluation",
            save_confusion_matrix=True, save_predictions_csv=True, save_f1_score=True,
            labels_dict=labels_dict_mapped
        )
    else:
        # ========== K-Fold completo su tutti i dati ==========
        skf = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=seed_used)
        all_fold_scores = []
        fold_results = {'fold': [], 'best_f1_score': [], 'best_params': [], 'test_f1_score': []}
        for fold, (train_val_idx, test_idx) in enumerate(skf.split(X, Y_mapped)):
            X_train_val, X_test_fold = X[train_val_idx], X[test_idx]
            Y_train_val, Y_test_fold = Y_mapped[train_val_idx], Y_mapped[test_idx]
            train_val_indices_meta = [all_window_indices[i] for i in train_val_idx]
            test_indices_meta = [all_window_indices[i] for i in test_idx]
            train_val_consec = all_consecutivity[train_val_idx]
            test_consec = all_consecutivity[test_idx]

            split_idx = int(0.8 * len(X_train_val))
            X_train_fold, Y_train_fold = X_train_val[:split_idx], Y_train_val[:split_idx]
            X_val_fold,   Y_val_fold   = X_train_val[split_idx:], Y_train_val[split_idx:]
            train_indices_meta = train_val_indices_meta[:split_idx]
            val_indices_meta   = train_val_indices_meta[split_idx:]
            train_consec       = train_val_consec[:split_idx]
            val_consec         = train_val_consec[split_idx:]

            if ft_aug == 'alltrs':
                logger.info("Applico data augmentation al set di training (fold).")
                transformation_function = generate_composite_transform_function_simple([
                    noise_transform_vectorized, scaling_transform_vectorized, negate_transform_vectorized,
                    time_flip_transform_vectorized, intra_sensor_channel_shuffle_transform_vectorized,
                    time_warp_transform_improved, time_warp_transform_low_cost
                ])
                X_train_aug = transformation_function(X_train_fold)
                X_train_fold = np.concatenate([X_train_fold, X_train_aug], axis=0)
                Y_train_fold = np.concatenate([Y_train_fold, Y_train_fold.copy()], axis=0)
                train_indices_meta.extend(train_indices_meta.copy())
                train_consec = np.concatenate([train_consec, train_consec.copy()], axis=0)

            train_dataset = HARDataset(X_train_fold, Y_train_fold, train_indices_meta, train_consec)
            val_dataset   = HARDataset(X_val_fold,   Y_val_fold,   val_indices_meta,   val_consec)
            test_dataset  = HARDataset(X_test_fold,  Y_test_fold,  test_indices_meta,  test_consec)

            def objective_fold(trial):
                lr = trial.suggest_categorical('lr', [5e-4, 1e-3, 5e-3, 1e-2])
                batch_size = trial.suggest_categorical('batch_size', [2, 4])
                loss_type = trial.suggest_categorical('loss_type', ['focal', 'label_smoothing', 'weighted_ce', 'combined'])
                if loss_type == 'focal':
                    gamma = trial.suggest_categorical('gamma', [1.0, 2.0, 3.0])
                    criterion = FocalLoss(gamma=gamma, num_classes=num_classes, task_type='multi-class')
                elif loss_type == 'label_smoothing':
                    smoothing = trial.suggest_categorical('smoothing', [0.05, 0.1, 0.15, 0.2])
                    criterion = LabelSmoothingCrossEntropy(epsilon=smoothing)
                elif loss_type == 'combined':
                    gamma = trial.suggest_categorical('gamma_combined', [1.0, 2.0])
                    focal_weight = trial.suggest_categorical('focal_weight', [0.3, 0.5, 0.7])
                    criterion = CombinedLoss(focal_weight=focal_weight, ce_weight=1.0 - focal_weight,
                                             gamma=gamma, class_weights=class_weights_tensor, num_classes=num_classes)
                else:
                    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, collate_fn=collate_fn)
                val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=collate_fn)
                model = DeepConvLSTM()                            # <= from scratch
                model = configure_model_for_tuning(model, tuning_strategy, num_classes)
                best_f1, best_state = train_with_cm.train(
                    net=model, train_loader=train_loader, val_loader=val_loader,
                    exp_figures_dir=exp_root_figures, exp_reports_dir=exp_root_reports,
                    epochs=args.epochs, lr=lr, criterion=criterion, save_plots=False
                )
                trial.set_user_attr("best_state", best_state)
                return best_f1

            study = optuna.create_study(direction='maximize', pruner=MedianPruner())
            study.optimize(objective_fold, n_trials=args.n_trials)
            best_params_fold = study.best_params
            best_f1_fold = study.best_value
            best_state = study.best_trial.user_attrs["best_state"]
            logger.info(f"[Fold {fold+1}] Best params: {best_params_fold} | Val F1: {best_f1_fold:.4f}")

            model_final_fold = DeepConvLSTM()
            model_final_fold = configure_model_for_tuning(model_final_fold, tuning_strategy, num_classes)
            model_final_fold.load_state_dict(best_state, strict=False)
            test_loader = DataLoader(test_dataset, batch_size=best_params_fold['batch_size'], shuffle=False, drop_last=False, collate_fn=collate_fn)
            _, _, test_f1_fold = train_with_cm.evaluate_model(
                net=model_final_fold, test_loader=test_loader,
                exp_figures_dir=exp_root_figures, exp_reports_dir=exp_root_reports,
                figure_name=f"TEST_fold_{fold + 1}",
                save_confusion_matrix=True, save_predictions_csv=True, save_f1_score=True,
                labels_dict=labels_dict_mapped
            )
            model_path_fold = os.path.join(exp_root_models, f"model_fold_{fold + 1}.pkl")
            torch.save(model_final_fold.state_dict(), model_path_fold)
            fold_results['fold'].append(fold + 1)
            fold_results['best_f1_score'].append(best_f1_fold)
            fold_results['best_params'].append(best_params_fold)
            fold_results['test_f1_score'].append(test_f1_fold)
            all_fold_scores.append({'fold': fold + 1, 'params': best_params_fold, 'val_f1': best_f1_fold,
                                    'test_f1': test_f1_fold, 'model_path': model_path_fold})

        combine_kfold_confusion_matrices(
            exp_reports_dir=exp_root_reports, num_folds=args.k_folds,
            class_names=[labels_dict_mapped[i] for i in range(num_classes)], exp_figures_dir=exp_root_figures
        )
        best_fold = max(all_fold_scores, key=lambda x: x['val_f1'])
        best_hyperparameters = best_fold['params']
        logger.info(f"Migliori iperparametri globali (da Fold {best_fold['fold']}): {best_hyperparameters}")
        final_dataset = HARDataset(X, Y_mapped, all_window_indices, all_consecutivity)
        final_train_loader = DataLoader(final_dataset, batch_size=best_hyperparameters['batch_size'],
                                        shuffle=True, collate_fn=collate_fn)
        final_model = DeepConvLSTM()
        final_model = configure_model_for_tuning(final_model, tuning_strategy, num_classes)
        final_f1_train, final_best_state = train_with_cm.train(
            net=final_model, train_loader=final_train_loader, val_loader=None,
            exp_figures_dir=exp_root_figures, exp_reports_dir=exp_root_reports,
            epochs=args.epochs, lr=best_hyperparameters['lr'],
            figure_name="final_training_on_all_train_val_data", save_plots=False
        )
        final_model.load_state_dict(final_best_state, strict=False)
        final_model_path = os.path.join(exp_root_models, "final_model.pkl")
        torch.save(final_model.state_dict(), final_model_path)
        logger.info(f"Modello finale salvato in: {final_model_path}")
        final_eval_loader = DataLoader(final_dataset, batch_size=best_hyperparameters['batch_size'],
                                       shuffle=False, collate_fn=collate_fn)
        train_with_cm.evaluate_model(
            net=final_model, test_loader=final_eval_loader,
            exp_figures_dir=exp_root_figures, exp_reports_dir=exp_root_reports,
            figure_name="FINAL_training_evaluation",
            save_confusion_matrix=True, save_predictions_csv=True, save_f1_score=True,
            labels_dict=labels_dict_mapped
        )

# ====== Argomenti CLI ======
def parse_args():
    p = argparse.ArgumentParser(description="Esegui esperimenti FROM-SCRATCH con stesso pre-processing di run_har_experiment.py")
    p.add_argument('--toy', type=str, required=True, choices=['ball', 'car', 'doll', 'spoon', 'elephant'],
                   help='SUPSI ADOS Dataset da usare.')
    # Scenari da eseguire. 'all' abilita le 4 combinazioni richieste.
    p.add_argument('--scenarios', nargs='+', default=['all'], choices=['none', 'norm', 'aug', 'both', 'all'],
                   help='Scenari: none | norm | aug | both | all')
    # Strategie di tuning da provare (di default entrambe)
    p.add_argument('--tuning-strategies', nargs='+', default=['lp', 'fft'], choices=['lp', 'fft'],
                   help='Strategie: lp (linear probing) e/o fft (full fine-tuning)')
    p.add_argument('--holdout-kids', nargs='*', type=int, default=None, help='Lista di ID bambini per hold-out test set.')
    p.add_argument('--k-folds', type=int, default=3, help='Numero di fold per la cross-validation.')
    p.add_argument('--epochs', type=int, default=100, help='Numero di epoche.')
    p.add_argument('--n-trials', type=int, default=50, help='Numero di trial Optuna.')
    p.add_argument('--filtered-indices-pkl', type=str, required=True,
                   help='Percorso al PKL generato da zupt_analysis.py con windows_keys_to_keep o windows_indices_to_keep.')
    return p.parse_args()

def main():
    args = parse_args()

    # ===== RICAVA IL SEED DAL PICKLE =====
    target_actions = None
    seed_used = 42
    npz_path = None
    
    if args.filtered_indices_pkl:
        try:
            with open(args.filtered_indices_pkl, 'rb') as f:
                idx_pack = pickle.load(f)
            target_actions = idx_pack.get('target_actions', None)
            seed_used = idx_pack.get('processing_info', {}).get('seed', seed_used)
            
            # Costruisci il path del NPZ dalla directory del pickle
            pkl_dir = os.path.dirname(args.filtered_indices_pkl)
            npz_filename = f'{args.toy}_processed_windows_seed{seed_used}.npz'
            npz_path = os.path.join(pkl_dir, npz_filename)
            
        except Exception as e:
            logger.warning(f"Errore nel caricamento del pickle: {e}")
            
    if seed_used is None:
        seed_used = 42  # fallback finale

    if npz_path is None or not os.path.exists(npz_path):
        raise FileNotFoundError(f"File NPZ non trovato: {npz_path}. Assicurati di aver eseguito zupt_analysis_2.py prima.")

    # ===== SETUP RIPRODUCIBILITÀ =====
    np.random.seed(seed_used)
    random.seed(seed_used)
    torch.manual_seed(seed_used)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_used)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger.info(f"Caricamento finestre pre-processate da: {npz_path}")
    logger.info(f"Target actions: {target_actions}")
    logger.info(f"Seed utilizzato: {seed_used}")


    # ===== Espandi scenari =====
    scenarios = args.scenarios
    if 'all' in scenarios:
        scenarios = ['none', 'norm', 'aug', 'both']
    scenario_to_cfg = {
        'none': ('none', 'none'),
        'norm': ('meanstdonft', 'none'),
        'aug' : ('none', 'alltrs'),
        'both': ('meanstdonft', 'alltrs')
    }

    # ===== Esegui tutte le combinazioni =====
    for ts in args.tuning_strategies:
        for sc in scenarios:
            ft_norm, ft_aug = scenario_to_cfg[sc]
            run_single_experiment(args=args, seed_used=seed_used, target_actions=target_actions,
                                  ft_norm=ft_norm, ft_aug=ft_aug, tuning_strategy=ts)

if __name__ == "__main__":
    main()
