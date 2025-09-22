# === run_har_from_scratch.py ===
# Questo script riproduce *lo stesso identico* pre-processing e la stessa pipeline
# di run_har_experiment.py, ma addestra i modelli **da zero** (no pesi pre-train).
# Le prove che può eseguire automaticamente sono:
#   1) from scratch "puro":        no norm, no aug   (sia LP che FFT)
#   2) solo normalizzazione:       mean/std sul training set (meanstdonft), no aug
#   3) solo augmentation:          aug sul training set (alltrs), no norm
#   4) entrambe:                   norm + aug
# Per ogni scenario puoi scegliere di eseguire LP (linear probing) e/o FFT (full fine-tuning).
#



import os
import argparse
import pickle
import random
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

# Optuna
import optuna
from optuna.pruners import MedianPruner

# Moduli progetto (identici allo script originale)
from app_config import REPORTS_DIR, FIGURES_DIR, MODELS_DIR
import sliding_window_on_data
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

# ====== Utility grafici / logging distribuzioni (copiati dallo script originale) ======
import matplotlib.pyplot as plt
from collections import Counter

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

# ===== Sanity check e bilanciamento hold-out (copiati) =====
def sanity_check_window_alignment(Y, window_meta, keep_idx=None, phase="check", sample_size=1000, strict=True):
    n = len(window_meta)
    if len(Y) != n:
        msg = f"[{phase}] Lunghezze diverse: len(Y)={len(Y)} vs len(meta)={n}"
        if strict: raise AssertionError(msg)
        logger.error(msg); return False
    idxs = np.arange(n)
    if sample_size is not None and n > sample_size:
        rng = np.random.default_rng(0)
        idxs = rng.choice(n, size=sample_size, replace=False)
    problems = 0
    for pos in idxs:
        meta = window_meta[pos]
        y_val = int(Y[pos])
        if int(meta['action_id']) != y_val:
            logger.error(f"[{phase}] Mismatch action_id @pos={pos}: Y={y_val}, meta.action_id={meta['action_id']}")
            problems += 1
            if strict: break
        rows = meta.get('row_indices', None)
        if rows is not None and len(rows) > 1:
            rows = np.asarray(rows)
            if not np.all(np.diff(rows) == 1):
                diffs_unique = np.unique(np.diff(rows))
                logger.warning(f"[{phase}] row_indices non contigui @pos={pos}: diffs unici {diffs_unique[:5]}")
        if keep_idx is not None:
            if int(meta['global_window_id']) != int(keep_idx[pos]):
                logger.error(f"[{phase}] Mismatch global_window_id @pos={pos}: meta={meta['global_window_id']} vs keep_idx={keep_idx[pos]}")
                problems += 1
                if strict: break
    if problems == 0:
        logger.info(f"[{phase}] Sanity check OK su {len(idxs)} finestre (n={n}).")
        return True
    if strict: raise AssertionError(f"[{phase}] Trovate {problems} incongruenze (vedi log).")
    return False

def balance_holdout_classes(X_train, Y_train, X_test, Y_test,
                            all_window_indices_train, all_window_indices_test,
                            all_consecutivity_train, all_consecutivity_test,
                            percentage_to_move=0.15, random_seed=42):
    logger.info("=== INIZIO BILANCIAMENTO CLASSI HOLDOUT TEST ===")
    train_class_distribution = Counter(Y_train)
    test_class_distribution = Counter(Y_test)
    all_classes = set(Y_train)
    missing_classes = [cls for cls in all_classes if cls not in test_class_distribution]
    logger.info(f"Classi mancanti nel test: {missing_classes}")
    if not missing_classes:
        logger.info("Tutte le classi sono già presenti nel test. Nessun bilanciamento necessario.")
        return (X_train, Y_train, X_test, Y_test,
                all_window_indices_train, all_window_indices_test,
                all_consecutivity_train, all_consecutivity_test)
    windows_to_move = {}
    for missing_class in missing_classes:
        total_windows = train_class_distribution[missing_class]
        windows_to_move[missing_class] = max(1, int(total_windows * percentage_to_move))
    np.random.seed(random_seed)
    indices_to_move = []
    for missing_class in missing_classes:
        class_indices = np.where(Y_train == missing_class)[0]
        n_to_move = windows_to_move[missing_class]
        if n_to_move > 0 and len(class_indices) >= n_to_move:
            selected_indices = np.random.choice(class_indices, size=n_to_move, replace=False)
            indices_to_move.extend(selected_indices)
        else:
            indices_to_move.extend(class_indices)
    indices_to_move = np.array(indices_to_move, dtype=int)
    if len(indices_to_move) == 0:
        logger.info("Nessuna finestra da spostare.")
        return (X_train, Y_train, X_test, Y_test,
                all_window_indices_train, all_window_indices_test,
                all_consecutivity_train, all_consecutivity_test)
    X_to_move = X_train[indices_to_move]
    Y_to_move = Y_train[indices_to_move]
    indices_to_move_meta = [all_window_indices_train[i] for i in indices_to_move]
    consec_to_move = all_consecutivity_train[indices_to_move]
    mask_keep = np.ones(len(X_train), dtype=bool)
    mask_keep[indices_to_move] = False
    X_train_updated = X_train[mask_keep]
    Y_train_updated = Y_train[mask_keep]
    all_window_indices_train_array = np.array(all_window_indices_train, dtype=object)
    indices_train_updated = all_window_indices_train_array[mask_keep].tolist()
    consec_train_updated = all_consecutivity_train[mask_keep]
    X_test_updated = np.concatenate([X_test, X_to_move], axis=0)
    Y_test_updated = np.concatenate([Y_test, Y_to_move], axis=0)
    indices_test_updated = all_window_indices_test + indices_to_move_meta
    consec_test_updated = np.concatenate([all_consecutivity_test, consec_to_move], axis=0)
    return (X_train_updated, Y_train_updated, X_test_updated, Y_test_updated,
            indices_train_updated, indices_test_updated, consec_train_updated, consec_test_updated)

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
def run_single_experiment(args, df_toy, seed_used, target_actions, ft_norm, ft_aug, tuning_strategy):
    # ========== Setup nome esperimento e cartelle ==========
    exp_name = f"FS_T{args.toy}_FTN{ft_norm}_FTA{ft_aug}_TS{tuning_strategy}"
    project_root = os.path.dirname(os.path.abspath(__file__))
    logs_exp_dir = os.path.join(project_root, "logs", exp_name)
    os.makedirs(logs_exp_dir, exist_ok=True)
    setup_logging(f"{exp_name}/seed_{seed_used}")
    exp_root_models  = os.path.join(MODELS_DIR,  exp_name, f"seed_{seed_used}")
    exp_root_figures = os.path.join(FIGURES_DIR, exp_name, f"seed_{seed_used}")
    exp_root_reports = os.path.join(REPORTS_DIR, exp_name, f"seed_{seed_used}")
    for p in [exp_root_models, exp_root_figures, exp_root_reports]:
        os.makedirs(p, exist_ok=True)
    logger.info(f"===== INIZIO ESPERIMENTO (FROM-SCRATCH): {exp_name} =====")
    logger.info(f"Seed: {seed_used}")
    logger.info(f"Cartelle: {exp_root_models}, {exp_root_figures}, {exp_root_reports}")

    # ========== Hold-out vs KFold ==========
    if args.holdout_kids:
        logger.info(f"Separazione hold-out kids: {args.holdout_kids}")
        df_test_holdout = df_toy[df_toy['kid_id'].isin(args.holdout_kids)]
        df_train_val = df_toy[~df_toy['kid_id'].isin(args.holdout_kids)].copy()
    else:
        logger.info("Nessun hold-out kids specificato: userò K-Fold su tutto il dataset.")
        df_train_val = df_toy.copy()
        df_test_holdout = None

    # ========== Normalizzazione a livello DataFrame ==========
    mean, std = None, None
    if ft_norm == 'meanstdonft':
        logger.info("Normalizzazione 'meanstdonft' (calcolata su df_train_val).")
        mean, std = normalization.compute_dataframe_mean_std(df_train_val)
    elif ft_norm == 'none':
        logger.info("Nessuna normalizzazione.")
    else:
        raise ValueError(f"ft_norm non supportata in from-scratch: {ft_norm}")

    if mean is not None and std is not None:
        df_train_val_norm = normalization.normalize_dataframe_mean_std(df_train_val, mean, std)
        if df_test_holdout is not None:
            df_test_holdout_norm = normalization.normalize_dataframe_mean_std(df_test_holdout, mean, std)
        else:
            df_test_holdout_norm = None 
    else:
        df_train_val_norm = df_train_val
        df_test_holdout_norm = df_test_holdout if df_test_holdout is not None else None

    if target_actions is not None:
        logger.info(f"Filtro target_actions PRIMA della sliding window: {target_actions}")
        logger.info(f"Azioni prima del filtro: {sorted(df_train_val_norm['action_id'].unique())}")
        df_train_val_norm = df_train_val_norm[df_train_val_norm['action_id'].isin(target_actions)]
        logger.info(f"Azioni dopo il filtro: {sorted(df_train_val_norm['action_id'].unique())}")
        if df_test_holdout_norm is not None:
            df_test_holdout_norm = df_test_holdout_norm[df_test_holdout_norm['action_id'].isin(target_actions)]

    # ========== Sliding window su train/val  ==========
    data_path = "C:\\\\codes\\\\HumanActivityRecognition\\\\data\\\\downstream_data"
    temp_action_dir = os.path.join(data_path, "temp_actions_fromscratch")
    os.makedirs(temp_action_dir, exist_ok=True)

    X_list, Y_list, all_consecutivity, all_windows_metadata_raw, kid_action_counts = [], [], [], [], defaultdict(dict)
    for action_id in sorted(df_train_val_norm['action_id'].unique()):
        df_action = df_train_val_norm[df_train_val_norm['action_id'] == action_id]
        temp_path = os.path.join(temp_action_dir, f'df_{args.toy}_action_{action_id}.csv')
        df_action.to_csv(temp_path, index=False)
        X_w, Y_w, kid_dict, is_consecutive, window_metadata = sliding_window_on_data.new_process_csv(
            temp_path, 9, 100, 50
        )
        X_list.append(X_w); Y_list.append(Y_w); all_consecutivity.extend(is_consecutive)
        all_windows_metadata_raw.extend(window_metadata)
        kid_action_counts[f"{args.toy}_action_{action_id}"] = kid_dict

    # pulizia temp
    try:
        import shutil
        shutil.rmtree(temp_action_dir)
    except Exception as e:
        logger.warning(f"Impossibile rimuovere temp_action_dir: {e}")

    X = np.concatenate(X_list, axis=0)
    Y = np.concatenate(Y_list, axis=0).flatten()
    all_consecutivity = np.array(all_consecutivity)

    all_window_indices = []
    for i, meta in enumerate(all_windows_metadata_raw):
        window_key = (meta.get('original_file', 'unknown'), tuple(meta.get('row_indices', [])))
        all_window_indices.append({
            'global_window_id': i,
            'action_id': meta['action_id'],
            'kid_id': meta.get('kid_id', 'unknown'),
            'original_file': meta.get('original_file', 'unknown'),
            'row_indices': meta['row_indices'],
            'window_key': window_key
        })
    logger.info(f"Finestre totali generate: {len(X)} | Classi: {sorted(np.unique(Y))}")

    sanity_check_window_alignment(Y=Y, window_meta=all_window_indices, keep_idx=None, phase="pre-seed", sample_size=1000, strict=True)
    log_and_plot_distribution(Y=Y, title_prefix="Distribuzione Iniziale (Prima del Filtro Seed)",
                              toy_name=args.toy, toy_mapping=mapping_activity.get_toy_mapping(args.toy.upper()),
                              save_dir=exp_root_figures)

    # ========== Filtro seed-specific dal PKL (finestre da mantenere) ==========
    with open(args.filtered_indices_pkl, 'rb') as f:
        idx_pack = pickle.load(f)
    windows_keys_to_keep = idx_pack.get('windows_keys_to_keep', None)
    keep_idx_fallback = idx_pack.get('windows_indices_to_keep', None)
    if windows_keys_to_keep is not None:
        logger.info("Uso chiavi stabili per il matching finestre.")
        key_to_local_index = {meta['window_key']: i for i, meta in enumerate(all_window_indices)}
        keep_idx = []
        missing_keys_count = 0
        for key_to_find in windows_keys_to_keep:
            if key_to_find in key_to_local_index:
                keep_idx.append(key_to_local_index[key_to_find])
            else:
                missing_keys_count += 1
        keep_idx = np.array(keep_idx, dtype=int)
        if missing_keys_count > 0:
            logger.error(f"{missing_keys_count} chiavi dal pickle non trovate tra le finestre generate.")
    elif keep_idx_fallback is not None:
        logger.warning("Fallback: uso indici numerici dal pickle (meno robusto).")
        keep_idx = np.array(keep_idx_fallback, dtype=int)
    else:
        raise KeyError("Nel PKL non trovo né 'windows_keys_to_keep' né 'windows_indices_to_keep'.")

    unique_keep_idx = np.unique(keep_idx)
    if len(unique_keep_idx) != len(keep_idx):
        seen = set()
        keep_idx = np.array([i for i in keep_idx if not (i in seen or seen.add(i))], dtype=int)

    X = X[keep_idx]; Y = Y[keep_idx]; all_consecutivity = all_consecutivity[keep_idx]
    all_window_indices = [all_window_indices[i] for i in keep_idx]
    logger.info(f"Filtro seed applicato: finestre tenute = {len(X)} | Classi: {sorted(np.unique(Y))}")
    sanity_check_window_alignment(Y=Y, window_meta=all_window_indices, keep_idx=keep_idx, phase="post-seed", sample_size=1000, strict=True)
    log_and_plot_distribution(Y=Y, title_prefix="Distribuzione Dopo Filtro Seed",
                              toy_name=args.toy, toy_mapping=mapping_activity.get_toy_mapping(args.toy.upper()),
                              save_dir=exp_root_figures)

    # ========== Remapping etichette e class weights ==========
    unique_labels = np.unique(Y)
    label_mapping = {label: i for i, label in enumerate(unique_labels)}
    Y_mapped = np.array([label_mapping[y] for y in Y])
    num_classes = len(unique_labels)

    toy_mapping_orig = mapping_activity.get_toy_mapping(args.toy.upper())
    labels_dict_mapped = {}
    for mapped_id in range(num_classes):
        original_id = None
        for orig, mapped in label_mapping.items():
            if mapped == mapped_id:
                original_id = orig; break
        if original_id is not None:
            class_name = toy_mapping_orig["original_to_name"].get(int(original_id), f"Unknown_{original_id}")
            labels_dict_mapped[mapped_id] = class_name
        else:
            labels_dict_mapped[mapped_id] = f"Class_{mapped_id}"

    class_weights = compute_class_weight('balanced', classes=np.unique(Y_mapped), y=Y_mapped)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
    logger.info(f"Pesi classi: {dict(zip(np.unique(Y_mapped), class_weights))}")

    # ========== Hold-out branch ==========
    if df_test_holdout is not None:
        # Sliding window su holdout (con la *stessa* normalizzazione se presente)
        X_test_list, Y_test_list = [], []
        all_window_indices_test, all_consecutivity_test = [], []
        data_path = "C:\\\\codes\\\\HumanActivityRecognition\\\\data\\\\downstream_data"
        temp_action_dir_test = os.path.join(data_path, "temp_actions_test_fromscratch")
        os.makedirs(temp_action_dir_test, exist_ok=True)
        for action_id in sorted(df_test_holdout_norm['action_id'].unique()):
            df_action_test = df_test_holdout_norm[df_test_holdout_norm['action_id'] == action_id]
            temp_path_test = os.path.join(temp_action_dir_test, f'df_test_{args.toy}_action_{action_id}.csv')
            df_action_test.to_csv(temp_path_test, index=False)
            X_w_te, Y_w_te, _, is_consec_te, win_idx_te = sliding_window_on_data.new_process_csv(
                temp_path_test, 9, 100, 50
            )
            X_test_list.append(X_w_te); Y_test_list.append(Y_w_te)
            all_consecutivity_test.extend(is_consec_te)
            for i in range(len(X_w_te)):
                window_key = (win_idx_te[i].get('original_file', 'unknown'), tuple(win_idx_te[i].get('row_indices', [])))
                all_window_indices_test.append({
                    'global_window_id': len(all_window_indices_test),
                    'action_id': action_id,
                    'kid_id': win_idx_te[i].get('kid_id', 'unknown'),
                    'original_file': win_idx_te[i].get('original_file', 'unknown'),
                    'row_indices': win_idx_te[i]['row_indices'],
                    'window_key': window_key
                })
        try:
            import shutil
            shutil.rmtree(temp_action_dir_test)
        except Exception as e:
            logger.warning(f"Impossibile rimuovere temp_action_dir_test: {e}")
        X_test_final = np.concatenate(X_test_list, axis=0)
        Y_test_final = np.concatenate(Y_test_list, axis=0).flatten()
        all_consecutivity_test = np.array(all_consecutivity_test)
        sanity_check_window_alignment(Y=Y_test_final, window_meta=all_window_indices_test, keep_idx=None,
                                      phase="holdout-test", sample_size=1000, strict=True)
        Y_test_final_mapped = np.array([label_mapping[y] for y in Y_test_final])

        # Bilanciamento classi holdout (opzionale come script originale)
        (X, Y_mapped, X_test_final, Y_test_mapped_balanced,
         all_window_indices, all_window_indices_test,
         all_consecutivity, all_consecutivity_test) = balance_holdout_classes(
            X_train=X, Y_train=Y_mapped, X_test=X_test_final, Y_test=Y_test_final_mapped,
            all_window_indices_train=all_window_indices, all_window_indices_test=all_window_indices_test,
            all_consecutivity_train=all_consecutivity, all_consecutivity_test=all_consecutivity_test,
            percentage_to_move=0.15, random_seed=seed_used
        )

        # Log distribuzioni aggiornate
        inv_label_mapping = {v: k for k, v in label_mapping.items()}
        Y_balanced_orig = np.array([inv_label_mapping[y] for y in Y_mapped])
        log_and_plot_distribution(Y=Y_balanced_orig, title_prefix="Training Set Dopo Bilanciamento Holdout",
                                  toy_name=args.toy, toy_mapping=toy_mapping_orig, save_dir=exp_root_figures)
        Y_test_final_orig = np.array([inv_label_mapping[y] for y in Y_test_mapped_balanced])
        log_and_plot_distribution(Y=Y_test_final_orig, title_prefix="Holdout Test Set Dopo Bilanciamento",
                                  toy_name=args.toy, toy_mapping=toy_mapping_orig, save_dir=exp_root_figures)

        # ====== KFold per HP, test interno per fold e training finale su tutto ======
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
        test_dataset_final = HARDataset(X_test_final, Y_test_mapped_balanced, window_indices=all_window_indices_test,
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

    # ===== inizializzazione/seed  =====
    seed_used = 42
    target_actions = None
    try:
        with open(args.filtered_indices_pkl, 'rb') as f:
            idx_pack = pickle.load(f)
        target_actions = idx_pack.get('target_actions', None)
        seed_used = idx_pack.get('processing_info', {}).get('seed', seed_used)
        if seed_used is None:
            import re
            match = re.search(r'seed[_\-]?(\d+)', args.filtered_indices_pkl)
            if match: seed_used = int(match.group(1))
    except Exception as e:
        pass
    if seed_used is None: seed_used = 42

    np.random.seed(seed_used); random.seed(seed_used); torch.manual_seed(seed_used)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed_used)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ===== Caricamento dati unificato + segment id  =====
    data_path = "C:\\\\codes\\\\HumanActivityRecognition\\\\data\\\\downstream_data"
    df_toy, toy_mapping = load_and_preprocess_data_unified(toy_name=args.toy, data_path=data_path, target_actions=target_actions)
    df_toy = add_consecutive_segment_id(df_toy)

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
            run_single_experiment(args=args, df_toy=df_toy, seed_used=seed_used, target_actions=target_actions,
                                  ft_norm=ft_norm, ft_aug=ft_aug, tuning_strategy=ts)

if __name__ == "__main__":
    main()
