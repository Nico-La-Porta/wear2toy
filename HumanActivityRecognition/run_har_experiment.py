# === run_har_experiment.py ===
# - Questo script assume che sia stato già eseguito lo script di (zupt_analysis.py)
#   e che per ogni seed si disponga del file:
#   zupt_preprocessing_output/<toy>/seed_<SEED>/<toy>_filtered_indices_seed<SEED>.pkl
#   contenente "windows_indices_to_keep" (indici GLOBALI rispetto alle finestre generate sull’intero dataset).
#
# - In questo script:
#   1) Vengono caricati TUTTI i CSV del giocattolo scelto da riga di comando.
#   2) Viene applicata la sliding window su tutto il dataset -> ottenendo X_all, Y_all, metadata_all.
#   4) Vengono caricati gli indici seed-specific dal pickle -> selezionando X_sel, Y_sel, meta_sel.
#   5) (Opzionale) Split hold-out a livello di finestre con meta_sel['kid_id'].
#   6) Normalizzazione SULLE FINESTRE (dopo la selezione):
#        - meanstdonpt: usa statistiche del pre-training.
#        - meanstdonft: calcola mean/std SOLO sul training e applica al test.
#        - none: niente.
#   7) K-fold + Optuna come prima, training/evaluate come train_with_cm.py


from email import parser
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


def parse_args():
    """
    Analizza gli argomenti della riga di comando per configurare l'esperimento.
    """
    parser = argparse.ArgumentParser(description="Esegui esperimenti di Wear2Toy in modo configurabile.")

    parser.add_argument('--toy', type=str, required=True, choices=['ball', 'car', 'doll', 'spoon', 'elephant'],
                        help='SUPSI ADOS Dataset da usare per il fine-tuning.')

    # Argomenti per il Pre-Training (PT)
    parser.add_argument('--pt-norm', type=str, default='none', choices=['none', 'meanstd'],
                        help='Normalizzazione usata durante il pre-training per selezionare il modello corretto.')
    parser.add_argument('--pt-aug', type=str, default='none', choices=['none', 'alltrs'],
                        help='Data augmentation usata durante il pre-training per selezionare il modello corretto.')

    # Argomenti per il Fine-Tuning (FT)
    parser.add_argument('--ft-norm', type=str, default='none', choices=['none', 'meanstdonpt', 'meanstdonft'],
                        help='Metodo di normalizzazione per il fine-tuning. "meanstdonpt" usa le statistiche del pre-training, "meanstdonft" calcola nuove statistiche.')
    parser.add_argument('--ft-aug', type=str, default='none', choices=['none', 'alltrs'],
                        help='Data augmentation da applicare durante il fine-tuning.')
    
    #Strategia di Tuning
    parser.add_argument('--tuning-strategy', type=str, default='fft', choices=['lp', 'fft'],
                        help='Strategia di fine-tuning: "lp" (linear probing con testa sequenziale) o "fft" (full fine-tuning con testa sequenziale).')
    parser.add_argument('--holdout-kids', nargs='*', type=int, default=None, help='Lista di ID di bambini da usare come hold-out test set.')
    # Parametri generali dell'esperimento
    parser.add_argument('--k-folds', type=int, default=3, help='Numero di fold per la cross-validation.')
    parser.add_argument('--epochs', type=int, default=100, help='Numero di epoche di training.')
    parser.add_argument('--n-trials', type=int, default=50, help='Numero di trial per la ricerca iperparametri con Optuna.')

    parser.add_argument(
       '--filtered-indices-pkl', type=str, default=None,
       help=("Percorso al file .pkl salvato da zupt_analysis.py con "
              "gli indici globali delle finestre da mantenere (seed-specific). "
              "Esempio: zupt_preprocessing_output/<toy>/seed_42/<toy>_filtered_indices_seed42.pkl")
   )

    return parser.parse_args()


def get_pretrained_model_path(pt_norm, pt_aug):
    """
    Restituisce il path del modello pre-addestrato corretto in base alla configurazione.
    """
    base_path = r'C:\codes\HumanActivityRecognition\models'
    
    if pt_norm == 'meanstd' and pt_aug == 'alltrs':
        return os.path.join(base_path, 'best_model_dl_norm_mean_std_and_aug.pkl')
    elif pt_norm == 'meanstd' and pt_aug == 'none':
        return os.path.join(base_path, 'best_model_dl_norm_mean_std_without_sampler.pkl')
    elif pt_norm == 'none' and pt_aug == 'alltrs':
        return os.path.join(base_path, 'best_model_dl_with_aug.pkl')
    elif pt_norm == 'none' and pt_aug == 'none':
        return os.path.join(base_path, 'best_model_dl_without_anything.pkl')
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





def balance_holdout_classes(X_train, Y_train, X_test, Y_test, 
                          all_window_indices_train, all_window_indices_test,
                          all_consecutivity_train, all_consecutivity_test,
                          percentage_to_move=0.15, random_seed=42):
    """
    Bilancia le classi nel test set spostando finestre dal training set
    quando alcune classi sono mancanti nel test set (holdout scenario).
    
    Args:
        X_train, Y_train: Dati di training
        X_test, Y_test: Dati di test (holdout)
        all_window_indices_train/test: Metadati delle finestre
        all_consecutivity_train/test: Informazioni di consecutività
        percentage_to_move: Percentuale di finestre da spostare per classe mancante
        random_seed: Seed per riproducibilità
    
    Returns:
        tuple: (X_train_updated, Y_train_updated, X_test_updated, Y_test_updated,
                indices_train_updated, indices_test_updated, 
                consec_train_updated, consec_test_updated)
    """
    logger.info("=== INIZIO BILANCIAMENTO CLASSI HOLDOUT TEST ===")
    
    # Conta distribuzione classi
    train_class_distribution = Counter(Y_train)
    test_class_distribution = Counter(Y_test)
    
    # Trova classi mancanti nel test
    all_classes = set(Y_train)
    missing_classes = [cls for cls in all_classes if cls not in test_class_distribution]
    
    logger.info(f"Classi totali nel training: {sorted(all_classes)}")
    logger.info(f"Classi presenti nel test: {sorted(test_class_distribution.keys())}")
    logger.info(f"Classi mancanti nel test: {missing_classes}")
    
    if not missing_classes:
        logger.info("Tutte le classi sono già presenti nel test. Nessun bilanciamento necessario.")
        return (X_train, Y_train, X_test, Y_test, 
                all_window_indices_train, all_window_indices_test,
                all_consecutivity_train, all_consecutivity_test)
    
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
        class_indices = np.where(Y_train == missing_class)[0]
        
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
        return (X_train, Y_train, X_test, Y_test, 
                all_window_indices_train, all_window_indices_test,
                all_consecutivity_train, all_consecutivity_test)
    
    # Estrai finestre da spostare
    X_to_move = X_train[indices_to_move]
    Y_to_move = Y_train[indices_to_move]
    indices_to_move_meta = [all_window_indices_train[i] for i in indices_to_move]
    consec_to_move = all_consecutivity_train[indices_to_move]
    
    # Crea maschera per mantenere le finestre nel training
    mask_keep = np.ones(len(X_train), dtype=bool)
    mask_keep[indices_to_move] = False
    
    # Aggiorna training set (rimuove finestre spostate)
    X_train_updated = X_train[mask_keep]
    Y_train_updated = Y_train[mask_keep]
    all_window_indices_train_array = np.array(all_window_indices_train, dtype=object)
    indices_train_updated = all_window_indices_train_array[mask_keep].tolist()
    consec_train_updated = all_consecutivity_train[mask_keep]
    
    # Aggiorna test set (aggiunge finestre spostate)
    X_test_updated = np.concatenate([X_test, X_to_move], axis=0)
    Y_test_updated = np.concatenate([Y_test, Y_to_move], axis=0)
    indices_test_updated = all_window_indices_test + indices_to_move_meta
    consec_test_updated = np.concatenate([all_consecutivity_test, consec_to_move], axis=0)
    
    # Log risultati finali
    logger.info("=== RISULTATI BILANCIAMENTO ===")
    logger.info(f"Training set: {X_train.shape} -> {X_train_updated.shape}")
    logger.info(f"Test set: {X_test.shape} -> {X_test_updated.shape}")
    
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
    
    return (X_train_updated, Y_train_updated, X_test_updated, Y_test_updated,
            indices_train_updated, indices_test_updated, 
            consec_train_updated, consec_test_updated)


def sanity_check_window_alignment(
    Y,
    window_meta,
    keep_idx=None,
    phase="check",
    sample_size=1000,
    strict=True
):
    """
    Verifica coerenza tra etichette e metadati delle finestre.
    - Controlla lunghezze Y vs metadati.
    - Verifica che meta['action_id'] == Y per un campione di finestre (o tutte).
    - Controlla contiguità di meta['row_indices'].
    - Se keep_idx è fornito (post-filtro), verifica che meta['global_window_id'] == keep_idx[pos].

    Args:
        Y (np.ndarray/list): etichette per-finestra.
        window_meta (list[dict]): ciascun dict deve contenere 'action_id', 'row_indices', 'global_window_id'.
        keep_idx (np.ndarray|list|None): indici originali tenuti (post-filtro seed). Se None, skip di questo check.
        phase (str): etichetta per il log.
        sample_size (int|None): quante finestre controllare. None = tutte.
        strict (bool): se True, solleva AssertionError al primo errore bloccante.

    Returns:
        bool: True se i controlli passano, False altrimenti (se strict=False).
    """
    n = len(window_meta)
    if len(Y) != n:
        msg = f"[{phase}] Lunghezze diverse: len(Y)={len(Y)} vs len(meta)={n}"
        if strict: 
            raise AssertionError(msg)
        else:
            logger.error(msg)
            return False

    # Scegliamo un sottoinsieme deterministico per velocità, se grande
    idxs = np.arange(n)
    if sample_size is not None and n > sample_size:
        rng = np.random.default_rng(0)  # deterministico
        idxs = rng.choice(n, size=sample_size, replace=False)

    problems = 0
    for pos in idxs:
        meta = window_meta[pos]
        y_val = int(Y[pos])

        # 1) Y vs action_id
        if int(meta['action_id']) != y_val:
            logger.error(f"[{phase}] Mismatch action_id @pos={pos}: Y={y_val}, meta.action_id={meta['action_id']}")
            problems += 1
            if strict:
                break

        # 2) contiguità row_indices
        rows = meta.get('row_indices', None)
        if rows is not None and len(rows) > 1:
            rows = np.asarray(rows)
            if not np.all(np.diff(rows) == 1):
                # warning (non bloccante): può capitare su dati particolari
                diffs_unique = np.unique(np.diff(rows))
                logger.warning(f"[{phase}] row_indices non contigui @pos={pos}: diffs unici {diffs_unique[:5]}")

        # 3) global_window_id vs keep_idx (solo post-filtro seed)
        if keep_idx is not None:
            if int(meta['global_window_id']) != int(keep_idx[pos]):
                logger.error(f"[{phase}] Mismatch global_window_id @pos={pos}: meta={meta['global_window_id']} vs keep_idx={keep_idx[pos]}")
                problems += 1
                if strict:
                    break

    if problems == 0:
        logger.info(f"[{phase}] Sanity check OK su {len(idxs)} finestre (n={n}).")
        return True

    if strict:
        raise AssertionError(f"[{phase}] Trovate {problems} incongruenze (vedi log).")
    return False


def main(args):
    """
    Funzione principale che esegue l'esperimento.
    """

    data_path = "C:\\codes\\HumanActivityRecognition\\data\\downstream_data"



    # ===== Ricava il seed dal pickle =====
    target_actions = None
    seed_used = 42
    if args.filtered_indices_pkl:
        try:
            with open(args.filtered_indices_pkl, 'rb') as f:
                idx_pack = pickle.load(f)
            target_actions = idx_pack.get('target_actions', None)
            seed_used = idx_pack.get('processing_info', {}).get('seed', None)
            if seed_used is None:
                # fallback: estrai dal nome file se contiene "seed_xx"
                import re
                match = re.search(r'seed[_\-]?(\d+)', args.filtered_indices_pkl)
                if match:
                    seed_used = int(match.group(1))
        except Exception as e:
            logger.warning(f"Impossibile leggere seed dal pickle: {e}")
    if seed_used is None:
        seed_used = 42  # fallback finale


    if target_actions is None:
        logger.warning("ATTENZIONE: 'target_actions' non trovato nel pickle. Il caricamento dei dati potrebbe non essere coerente con lo script ZUPT.")

    logger.info(f"Target actions dal pickle: {target_actions}")

    df_toy, toy_mapping = load_and_preprocess_data_unified(
    toy_name=args.toy, 
    data_path=data_path,
    target_actions=target_actions 
    )
    
    logger.info(f"COLONNE SUBITO DOPO IL CARICAMENTO: {df_toy.columns.tolist()}")
    df_toy = add_consecutive_segment_id(df_toy)
    logger.info(f"Classi presenti dopo il caricamento unificato: {sorted(df_toy['action_id'].unique())}")
    
    
    
    
    
    # === Reproducibilità: fissa tutti i semi ===
    np.random.seed(seed_used)
    random.seed(seed_used)
    torch.manual_seed(seed_used)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_used)
    # cuDNN deterministico
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ===== Nome esperimento (senza seed) =====
    exp_name = (
        f"T{args.toy}_PTN{args.pt_norm}_PTA{args.pt_aug}_"
        f"FTN{args.ft_norm}_FTA{args.ft_aug}_TS{args.tuning_strategy}"
    )

    # ===== Cartelle root esperimento =====
    exp_root_models  = os.path.join(MODELS_DIR,  exp_name)
    exp_root_figures = os.path.join(FIGURES_DIR, exp_name)
    exp_root_reports = os.path.join(REPORTS_DIR, exp_name)
    os.makedirs(exp_root_models, exist_ok=True)
    os.makedirs(exp_root_figures, exist_ok=True)
    os.makedirs(exp_root_reports, exist_ok=True)

    # ===== Cartelle seed-specifiche =====
    seed_tag = f"seed_{seed_used}"
    exp_models_dir  = os.path.join(exp_root_models,  seed_tag)
    exp_figures_dir = os.path.join(exp_root_figures, seed_tag)
    exp_reports_dir = os.path.join(exp_root_reports, seed_tag)
    os.makedirs(exp_models_dir, exist_ok=True)
    os.makedirs(exp_figures_dir, exist_ok=True)
    os.makedirs(exp_reports_dir, exist_ok=True)
    

    project_root = os.path.dirname(os.path.abspath(__file__))
    logs_exp_dir = os.path.join(project_root, "logs", exp_name)
    os.makedirs(logs_exp_dir, exist_ok=True)
    setup_logging(f"{exp_name}/{seed_tag}")  # log separati per esperimento/seed

    # ===== Logging =====
    logger.info(f"===== INIZIO ESPERIMENTO: {exp_name} =====")
    logger.info(f"Seed corrente: {seed_used}")
    logger.info(f"Cartelle: {exp_models_dir}, {exp_figures_dir}, {exp_reports_dir}")

    


    # --- Hold-out set per bambini specifici --- 

    if args.holdout_kids:
        logger.info(f"Separazione dei bambini per il test hold-out: {args.holdout_kids}")
        df_test_holdout = df_toy[df_toy['kid_id'].isin(args.holdout_kids)]
        df_train_val = df_toy[~df_toy['kid_id'].isin(args.holdout_kids)]
        logger.info(f"Dimensioni Training/Validation set: {df_train_val.shape}")
        logger.info(f"Dimensioni Hold-out Test set: {df_test_holdout.shape}")
    else:
        logger.info("Nessun hold-out set specificato: Esecuzione K-Fold su tutto il dataset.")
        df_train_val = df_toy.copy()
        df_test_holdout = None

    # --- Normalizzazione ---
    df_normalized = pd.DataFrame()
    mean, std = None, None
    if args.ft_norm == 'meanstdonpt':
        logger.info("Normalizzazione con statistiche del pre-training.")
        mean_df, std_df = pd.read_csv(os.path.join(REPORTS_DIR, 'mean_trs.csv')), pd.read_csv(os.path.join(REPORTS_DIR, 'std_trs.csv'))
        mean, std = mean_df['mean'].values[1:-1], std_df['std'].values[1:-1]
        logger.info("Parametri di normalizzazione caricati dal pre-training")
        logger.debug(f"Mean shape: {mean.shape}, Std shape: {std.shape}")
    elif args.ft_norm == 'meanstdonft':
        logger.info("Normalizzazione con statistiche calcolate sul dataset di fine-tuning.")
        mean, std = normalization.compute_dataframe_mean_std(df_train_val)
    
    if mean is not None and std is not None:
        df_normalized = normalization.normalize_dataframe_mean_std(df_train_val, mean, std)
        if args.holdout_kids:
            df_test_holdout_normalized = normalization.normalize_dataframe_mean_std(df_test_holdout, mean, std)
    else:
        logger.info("Nessuna normalizzazione al fine tuning applicata.")
        df_normalized = df_train_val
        if args.holdout_kids:
            df_test_holdout_normalized = df_test_holdout
    
    logger.info(f"COLONNE DOPO LA NORMALIZZAZIONE: {df_normalized.columns.tolist()}")

    if target_actions is not None:
        logger.info(f"Filtro target_actions PRIMA della sliding window: {target_actions}")
        logger.info(f"Azioni prima del filtro: {sorted(df_normalized['action_id'].unique())}")
        df_normalized = df_normalized[df_normalized['action_id'].isin(target_actions)]
        logger.info(f"Azioni dopo il filtro: {sorted(df_normalized['action_id'].unique())}")
        logger.info(f"Righe rimanenti dopo filtro target_actions: {len(df_normalized)}")

    # --- Salvataggio file per azione e Sliding Window ---
    logger.info("Salvataggio file per azione e applicazione sliding window...")
    X_list, Y_list, all_consecutivity, all_windows_metadata_raw ,kid_action_counts = [], [], [], [], defaultdict(dict)
    
    temp_action_dir = os.path.join(data_path, "temp_actions")
    os.makedirs(temp_action_dir, exist_ok=True)

    for action_id in sorted(df_normalized['action_id'].unique()):
        df_action = df_normalized[df_normalized['action_id'] == action_id]
        temp_path = os.path.join(temp_action_dir, f'df_{args.toy}_action_{action_id}.csv')
        df_action.to_csv(temp_path, index=False)
        logger.debug(f"Salvato file temporaneo: {temp_path}")

        X_w, Y_w, kid_dict, is_consecutive, window_metadata = sliding_window_on_data.new_process_csv(
            temp_path, 9, 100, 50
        )
        X_list.append(X_w); Y_list.append(Y_w); all_consecutivity.extend(is_consecutive)
        all_windows_metadata_raw.extend(window_metadata)

        logger.info(f"Numero totale di finestre per l'azione {action_id}: {len(X_w)}")
        logger.info(f"Finestre consecutive per l'azione {action_id}: {sum(is_consecutive)}")
        logger.info(f"Finestre non consecutive per l'azione {action_id}: {len(is_consecutive) - sum(is_consecutive)}")
        logger.info(f"Numero  totale di finestre per l'azione {action_id}:{len(X_w)}")
        kid_action_counts[f"{args.toy}_action_{action_id}"] = kid_dict #dizionario principale per tenere traccia del numero di finestre per ogni bambino per ogni azione, ogni spoon_action è una chiave e il valore è un dizionario con il numero di finestre per ogni bambino
        logger.info(f"Azione {action_id}: finestre={len(X_w)}, consecutive={sum(is_consecutive)}") #per vedere quante finestre per ogni azione e per ogni bambino sono state elaborte 
            


    shutil.rmtree(temp_action_dir) # Pulisce la cartella temporanea

    X = np.concatenate(X_list, axis=0)
    Y = np.concatenate(Y_list, axis=0).flatten()
    all_consecutivity = np.array(all_consecutivity)

    all_window_indices = []
    for i, meta in enumerate(all_windows_metadata_raw):
        # 🔧 CHIAVE STABILE: (original_file, tuple(row_indices))
        window_key = (meta.get('original_file', 'unknown'), tuple(meta.get('row_indices', [])))
        
        all_window_indices.append({
            'global_window_id': i,
            'action_id': meta['action_id'],
            'kid_id': meta.get('kid_id', 'unknown'),  # Aggiungi kid_id
            'original_file': meta.get('original_file', 'unknown'),  # Aggiungi original_file
            'row_indices': meta['row_indices'],
            'window_key': window_key  # 🔧 CHIAVE STABILE
        })

    logger.info(f"Finestre totali generate: {len(X)}")
    logger.info(f"Azioni presenti nelle finestre: {sorted(np.unique(Y))}")


    logger.info("=== DEBUG CHIAVI FINESTRE ===")
    for i in range(min(5, len(all_window_indices))):
        meta = all_window_indices[i]
        logger.info(f"Finestra {i}: key={meta['window_key'][:2]}...{meta['window_key'][-2:]} action={meta['action_id']}")
    
    sanity_check_window_alignment(
    Y=Y,
    window_meta=all_window_indices,
    keep_idx=None,
    phase="pre-seed",
    sample_size=1000,
    strict=True
    )
    # Distribuzione prima del filtro seed
    log_and_plot_distribution(
        Y=Y,
        title_prefix="Distribuzione Iniziale (Prima del Filtro Seed)",
        toy_name=args.toy,
        toy_mapping=toy_mapping,
        save_dir=exp_figures_dir
    )

    # --- FILTRO FINESTRE DA PICKLE (SEED-SPECIFIC) ---
    if args.filtered_indices_pkl is None or not os.path.exists(args.filtered_indices_pkl):
        raise FileNotFoundError("Passa --filtered-indices-pkl con il file .pkl generato da zupt_analysis.py")
    with open(args.filtered_indices_pkl, 'rb') as f:
        idx_pack = pickle.load(f)

    # 🔧 CARICA LE CHIAVI STABILI invece degli indici numerici
    windows_keys_to_keep = idx_pack.get('windows_keys_to_keep', None)
    keep_idx_fallback = idx_pack.get('windows_indices_to_keep', None)

    if windows_keys_to_keep is not None:
        # 🔧 NUOVO METODO: Usa le chiavi stabili
        logger.info("Usando chiavi stabili per il matching delle finestre")
        
        # Crea mappa: window_key -> indice locale
        key_to_local_index = {meta['window_key']: i for i, meta in enumerate(all_window_indices)}
        
        # Trova gli indici corrispondenti alle chiavi da mantenere
        keep_idx = []
        missing_keys_count = 0
        
        for key_to_find in windows_keys_to_keep:
            if key_to_find in key_to_local_index:
                keep_idx.append(key_to_local_index[key_to_find])
            else:
                missing_keys_count += 1
        
        keep_idx = np.array(keep_idx, dtype=int)
        
        logger.info(f"Finestre totali generate localmente: {len(X)}")
        logger.info(f"Chiavi da mantenere (dal pickle): {len(windows_keys_to_keep)}")
        logger.info(f"Chiavi trovate e matchate: {len(keep_idx)}")
        
        if missing_keys_count > 0:
            logger.error(f"ATTENZIONE: {missing_keys_count} chiavi dal file pickle non sono state trovate tra le finestre generate!")
            logger.error("Questo indica ancora una discrepanza nella generazione dei dati iniziali. Controlla che i file CSV e i filtri siano identici.")
            
    elif keep_idx_fallback is not None:
        # 🔧 FALLBACK: Usa il metodo vecchio (indici numerici)
        logger.warning("Usando metodo fallback con indici numerici (potrebbe essere impreciso)")
        keep_idx = np.array(keep_idx_fallback, dtype=int)
        
        # Verifica range
        if keep_idx.min() < 0 or keep_idx.max() >= len(X):
            logger.error(f"ERRORE: Indici fuori range con metodo fallback!")
            logger.error(f"   Finestre generate: {len(X)}")
            logger.error(f"   Indici min/max: {keep_idx.min()}/{keep_idx.max()}")
            raise AssertionError("Indici keep fuori range - rigenera il pickle con la versione aggiornata")
            
    else:
        raise KeyError("Nel PKL non trovo né 'windows_keys_to_keep' né 'windows_indices_to_keep'")

    # Rimuovi duplicati se presenti
    unique_keep_idx = np.unique(keep_idx)
    if len(unique_keep_idx) != len(keep_idx):
        logger.warning("keep_idx ha duplicati — li rimuovo preservando l'ordine.")
        seen = set()
        keep_idx = np.array([i for i in keep_idx if not (i in seen or seen.add(i))], dtype=int)

    # Applica il filtro
    X = X[keep_idx]
    Y = Y[keep_idx]
    all_consecutivity = all_consecutivity[keep_idx]
    all_window_indices = [all_window_indices[i] for i in keep_idx]

    logger.info(f"Filtro seed applicato: finestre tenute = {len(X)}")
    logger.info(f"Azioni finali dopo filtro: {sorted(np.unique(Y))}")


    # --- CHECK 2: coerenza Y/meta DOPO il filtro seed (verifica anche global_window_id vs keep_idx) ---
    sanity_check_window_alignment(
        Y=Y,
        window_meta=all_window_indices,
        keep_idx=keep_idx,
        phase="post-seed",
        sample_size=1000,
        strict=True
    )
    # Distribuzione dopo filtro seed
    log_and_plot_distribution(
        Y=Y,
        title_prefix="Distribuzione Dopo Filtro Seed",
        toy_name=args.toy,
        toy_mapping=toy_mapping,
        save_dir=exp_figures_dir
    )

   

    unique_labels = np.unique(Y)
    label_mapping = {label: i for i, label in enumerate(unique_labels)}
    Y_mapped = np.array([label_mapping[y] for y in Y])
    num_classes = len(unique_labels)

    toy_mapping_orig = mapping_activity.get_toy_mapping(args.toy.upper())
    labels_dict_mapped = {}
    
    for mapped_id in range(num_classes):
        # Trova l'ID originale corrispondente a questo mapped_id
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
    

    # Calcola i pesi in base alla frequenza inversa delle classi.
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(Y_mapped),
        y=Y_mapped
    )
    # Converte i pesi in un tensore PyTorch, pronto per essere usato dalla loss.
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
    
    logger.info(f"Pesi delle classi calcolati: {dict(zip(np.unique(Y_mapped), class_weights))}")

    logger.info(f"Dati di training/validation pronti: X.shape={X.shape}, num_classes={num_classes}")
    pretrained_model_path = get_pretrained_model_path(args.pt_norm, args.pt_aug)
    logger.info(f"Utilizzo del modello pre-addestrato: {pretrained_model_path}")

    if args.holdout_kids:
        logger.info("AVVIO PROCEDURA: K-Fold su Train/Val set + Valutazione finale su Hold-out")

        
        logger.info("Preparazione del hold-out test...")

        # Il DataFrame df_test_holdout_normalized è già stato normalizzato correttamente all'inizio.
        # Ora applichiamo la sliding window su di esso.
        X_test_list, Y_test_list = [], []
        all_window_indices_test, all_consecutivity_test = [], []
        temp_action_dir_test = os.path.join(data_path, "temp_actions_test")
        os.makedirs(temp_action_dir_test, exist_ok=True)

        logger.info("Applicazione Sliding Window sul set di hold-out test...")
        for action_id in sorted(df_test_holdout_normalized['action_id'].unique()):
            df_action_test = df_test_holdout_normalized[df_test_holdout_normalized['action_id'] == action_id]
            temp_path_test = os.path.join(temp_action_dir_test, f'df_test_{args.toy}_action_{action_id}.csv')
            df_action_test.to_csv(temp_path_test, index=False)
            X_w_te, Y_w_te, _, is_consec_te, win_idx_te = sliding_window_on_data.new_process_csv(
                temp_path_test, 9, 100, 50
            )
            X_test_list.append(X_w_te); Y_test_list.append(Y_w_te)
            all_consecutivity_test.extend(is_consec_te)
            # metadati test
            for i in range(len(X_w_te)):
                # Crea la chiave stabile per questa finestra
                window_key = (win_idx_te[i].get('original_file', 'unknown'), tuple(win_idx_te[i].get('row_indices', [])))
                
                all_window_indices_test.append({
                    'global_window_id': len(all_window_indices_test),
                    'action_id': action_id,
                    'kid_id': win_idx_te[i].get('kid_id', 'unknown'),  # 🔧 Aggiungi kid_id
                    'original_file': win_idx_te[i].get('original_file', 'unknown'),  # 🔧 Aggiungi original_file
                    'row_indices': win_idx_te[i]['row_indices'],
                    'window_key': window_key  # 🔧 CHIAVE STABILE
                })

        shutil.rmtree(temp_action_dir_test)
        
        # Concatena i risultati per creare gli array finali del test set.
        X_test_final = np.concatenate(X_test_list, axis=0)
        Y_test_final = np.concatenate(Y_test_list, axis=0).flatten()
        all_consecutivity_test = np.array(all_consecutivity_test)

        # --- CHECK 3: coerenza sul test hold-out ---
        sanity_check_window_alignment(
            Y=Y_test_final,
            window_meta=all_window_indices_test,
            keep_idx=None,            # qui non c'è keep_idx
            phase="holdout-test",
            sample_size=1000,
            strict=True
        )
        logger.info(f"Dati di test finali pronti: X_test_final.shape={X_test_final.shape}")

        # Applica lo STESSO remapping di etichette usato per il training.
        Y_test_final_mapped = np.array([label_mapping[y] for y in Y_test_final])



        logger.info(f"Holdout test set prima del bilanciamento: X_test_holdout.shape={X_test_final.shape}")
        logger.info(f"Y_train dtypes: {Y_mapped.dtype}, Y_test dtypes: {Y_test_final_mapped.dtype}")
        logger.info(f"Y_train sample: {Y_mapped[:5]}, Y_test sample: {Y_test_final_mapped[:5]}")
        # APPLICA IL BILANCIAMENTO DELLE CLASSI
        (X_balanced, Y_balanced, X_test_final, Y_test_mapped_balanced,
        indices_balanced, indices_test_final, 
        consec_balanced, consec_test_final) = balance_holdout_classes(
            X_train=X, Y_train=Y_mapped, 
            X_test=X_test_final, Y_test=Y_test_final_mapped,
            all_window_indices_train=all_window_indices[:len(X)],  # <-- usa solo i metadati delle finestre di training
            all_window_indices_test=all_window_indices_test,
            all_consecutivity_train=all_consecutivity[:len(X)],
            all_consecutivity_test=all_consecutivity_test,
            percentage_to_move=0.15,
            random_seed=seed_used
        )
        
        # Aggiorna le variabili con i dati bilanciati
        X = X_balanced
        Y_mapped = Y_balanced
        all_window_indices = indices_balanced
        all_consecutivity = consec_balanced


        inv_label_mapping = {v: k for k, v in label_mapping.items()}
        Y_balanced_orig = np.array([inv_label_mapping[y] for y in Y_balanced])
        
        # Log delle distribuzioni aggiornate
        log_and_plot_distribution(
            Y=Y_balanced_orig, 
            title_prefix="Training Set Dopo Bilanciamento Holdout",
            toy_name=args.toy,
            toy_mapping=toy_mapping,
            save_dir=exp_figures_dir
        )

        Y_test_final_orig = np.array([inv_label_mapping[y] for y in Y_test_mapped_balanced])
        log_and_plot_distribution(
            Y=Y_test_final_orig,
            title_prefix="Holdout Test Set Dopo Bilanciamento", 
            toy_name=args.toy,
            toy_mapping=toy_mapping,
            save_dir=exp_figures_dir
        )
        
        skf = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=seed_used)
        all_fold_scores = []


        # Dizionario per i risultati
        fold_results = {
            'fold': [], 'best_f1_score': [], 'best_params': [], 'test_f1_score': []
        }

        

        for fold, (train_val_idx, test_idx) in enumerate(skf.split(X, Y_mapped)):
            
            # --- BLOCCO DI DEBUG DEGLI INDICI ---
            logger.info(f"\n===== FOLD {fold + 1}/{args.k_folds} | DEBUG DELLO SPLIT =====")
            logger.info(f"Train+Val indices totali: {len(train_val_idx)}")
            logger.info(f"Test indices totali: {len(test_idx)}")
            logger.debug(f"Primi 10 Train+Val indices: {train_val_idx[:10]}")
            logger.debug(f"Primi 10 Test indices: {test_idx[:10]}")
            
            overlap = set(train_val_idx).intersection(set(test_idx))
            if overlap: logger.error(f"ERRORE: SOVRAPPOSIZIONE TROVATA TRA TRAIN E TEST: {overlap}")
            else: logger.info("Controllo sovrapposizione: OK. Nessuna sovrapposizione.")
            
            test_window_ids = [all_window_indices[i]['global_window_id'] for i in test_idx]
            logger.info(f"Global IDs delle finestre nel set di test (primi 10): {test_window_ids[:10]}")
        

            X_train_val, X_test_fold = X[train_val_idx], X[test_idx]
            Y_train_val, Y_test_fold = Y_mapped[train_val_idx], Y_mapped[test_idx]

            train_val_indices_meta = [all_window_indices[i] for i in train_val_idx]
            test_indices_meta = [all_window_indices[i] for i in test_idx]
            train_val_consec = all_consecutivity[train_val_idx]
            test_consec = all_consecutivity[test_idx]
            
            split_idx = int(0.8 * len(X_train_val))
            X_train_fold, Y_train_fold = X_train_val[:split_idx], Y_train_val[:split_idx]
            X_val_fold, Y_val_fold = X_train_val[split_idx:], Y_train_val[split_idx:]
            
            train_indices_meta = train_val_indices_meta[:split_idx]
            val_indices_meta = train_val_indices_meta[split_idx:]
            train_consec = train_val_consec[:split_idx]
            val_consec = train_val_consec[split_idx:]
            
            if args.ft_aug == 'alltrs':
                logger.info("Applicazione data augmentation al set di training del fold.")
                transformation_function = generate_composite_transform_function_simple([
                    noise_transform_vectorized, scaling_transform_vectorized, negate_transform_vectorized,
                    time_flip_transform_vectorized, channel_shuffle_transform_vectorized,
                    time_warp_transform_improved, time_warp_transform_low_cost
                ])
                X_train_aug = transformation_function(X_train_fold)
                X_train_fold = np.concatenate([X_train_fold, X_train_aug], axis=0)
                Y_train_fold = np.concatenate([Y_train_fold, Y_train_fold.copy()], axis=0)
                train_indices_meta.extend(train_indices_meta.copy())
                train_consec = np.concatenate([train_consec, train_consec.copy()], axis=0)
            train_dataset = HARDataset(X_train_fold, Y_train_fold, train_indices_meta, train_consec)
            val_dataset = HARDataset(X_val_fold, Y_val_fold, val_indices_meta, val_consec)
            test_dataset = HARDataset(X_test_fold, Y_test_fold, test_indices_meta, test_consec)


            def objective_fold(trial):
                #lr = trial.suggest_categorical('lr', [1e-5, 1e-4, 1e-3]) VALIDATION LOSS PIATTA 
                lr = trial.suggest_categorical('lr', [5e-4, 1e-3, 5e-3, 1e-2]) 

                batch_size = trial.suggest_categorical('batch_size', [2, 4])
                loss_type = trial.suggest_categorical('loss_type', [
                    'focal', 'label_smoothing', 'weighted_ce', 'combined'
                ])
                criterion = None
                if loss_type == 'focal':
                    # Per FocalLoss, suggerisce un valore per gamma.
                    gamma = trial.suggest_categorical('gamma', [1.0, 2.0, 3.0])
                    # (Assicurati che FocalLoss sia importata correttamente)
                    criterion = FocalLoss(gamma=gamma, num_classes=num_classes, task_type='multi-class')
                    
                elif loss_type == 'label_smoothing':
                    # Per LabelSmoothing, suggerisce un valore di smoothing.
                    smoothing = trial.suggest_categorical('smoothing', [0.05, 0.1, 0.15, 0.2])
                    # (Assicurati che LabelSmoothingCrossEntropy sia importata)
                    criterion = LabelSmoothingCrossEntropy(epsilon=smoothing)
                    
                elif loss_type == 'combined':
                    # Per la loss combinata, suggerisce i parametri specifici.
                    gamma = trial.suggest_categorical('gamma_combined', [1.0, 2.0]) # Nome diverso per evitare conflitti
                    focal_weight = trial.suggest_categorical('focal_weight', [0.3, 0.5, 0.7])
                    criterion = CombinedLoss(
                        focal_weight=focal_weight,
                        ce_weight=1.0 - focal_weight,
                        gamma=gamma,
                        class_weights=class_weights_tensor, # Usa i pesi calcolati prima
                        num_classes=num_classes
                    )
                else:  # 'weighted_ce'
                    # Per la Cross Entropy pesata, usa i pesi calcolati prima.
                    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, collate_fn=collate_fn)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=collate_fn)
                
                model = DeepConvLSTM()
                model.load_state_dict(torch.load(pretrained_model_path, map_location='cpu'), strict=False)
                model = configure_model_for_tuning(model, args.tuning_strategy, num_classes)
                logger.info(f"TRIAL {trial.number}: lr={lr}, batch={batch_size}, loss={loss_type}")
                best_f1, best_state  = train_with_cm.train( net=model, 
                    train_loader=train_loader, 
                    val_loader=val_loader,
                    exp_figures_dir=exp_figures_dir,
                    exp_reports_dir=exp_reports_dir,
                    epochs=args.epochs, 
                    lr=lr,
                    criterion=criterion,
                    save_plots=False 
                )

                trial.set_user_attr("best_state", best_state)

                return best_f1
            
            # Ottimizzazione per questo fold
            logger.info(f"Inizio ottimizzazione iperparametri per fold {fold + 1}")
            study = optuna.create_study(direction='maximize', pruner=MedianPruner())
            study.optimize(objective_fold, n_trials=args.n_trials)
            
            best_params_fold = study.best_params
            best_f1_fold = study.best_value
            best_trial = study.best_trial
            best_state = best_trial.user_attrs["best_state"]
            logger.info(f"Migliori iperparametri per il fold {fold + 1}: {best_params_fold} (Val F1: {best_f1_fold:.4f})")

            model_final_fold = DeepConvLSTM()
            model_final_fold.load_state_dict(torch.load(pretrained_model_path, map_location='cpu'), strict=False)
            model_final_fold = configure_model_for_tuning(model_final_fold, args.tuning_strategy, num_classes)
            model_final_fold.load_state_dict(best_state, strict=False) 
            
            test_loader = DataLoader(test_dataset, batch_size=best_params_fold['batch_size'], shuffle=False, drop_last=False, collate_fn=collate_fn)

            logger.info(f"\n=== DEBUG DATALOADER FOLD {fold + 1} ===")
            logger.info(f"Test dataset size: {len(test_dataset)}")
            test_batches = len(test_loader)

            # CALCOLO CORRETTO delle finestre usate (con drop_last=False usa TUTTE)
            test_samples_used = len(test_dataset)

            logger.info(f"Test batches: {test_batches} (finestre usate: {test_samples_used}/{len(test_dataset)})")



            # --- BLOCCO DI DEBUG PRE-VALUTAZIONE ---
            logger.info(f"\n=== PRE-EVALUATE DEBUG (Fold {fold+1}) ===")
            logger.info(f"Test dataset size prima di evaluate: {len(test_dataset)}")
            logger.info(f"Test loader size prima di evaluate: {len(test_loader)}")
            if len(test_loader) > 0:
                for i, batch in enumerate(test_loader):
                    if i == 0:
                        inputs, targets, indices, _ = batch
                        logger.info(f"Primo batch di test - inputs shape: {inputs.shape}")
                        logger.info(f"Primo batch di test - targets shape: {targets.shape}")
                        logger.info(f"Primo batch di test - indices: {indices}")
                        break

                    
            _, _, test_f1_fold = train_with_cm.evaluate_model(
                model_final_fold, test_loader,
                figure_name=f"cm_test_fold_{fold + 1}_{exp_name}",
                save_confusion_matrix=True, 
                save_predictions_csv=True,
                save_f1_score=True,
                labels_dict=labels_dict_mapped,
                exp_figures_dir=exp_figures_dir,
                exp_reports_dir=exp_reports_dir
            )

            model_path_fold = os.path.join(exp_models_dir, f"best_model_fold_{fold+1}.pth")
            torch.save(model_final_fold.state_dict(), model_path_fold)

            fold_results['fold'].append(fold + 1)
            fold_results['best_f1_score'].append(best_f1_fold)
            fold_results['best_params'].append(best_params_fold)
            fold_results['test_f1_score'].append(test_f1_fold)
            
            all_fold_scores.append({
                'fold': fold + 1, 'params': best_params_fold, 'val_f1': best_f1_fold,
                'test_f1': test_f1_fold, 'model_path': model_path_fold
            })
            
            logger.info(f"Fold {fold + 1} completato - Test F1: {test_f1_fold:.4f}")

        logger.info("\n===== FASE FINALE: Training su tutto Train/Val e Valutazione su Hold-out =====")
        logger.info("Identificazione dei migliori iperparametri globali...")
        best_fold = max(all_fold_scores, key=lambda x: x['val_f1'])
        best_hyperparameters = best_fold['params']
        logger.info(f"Migliori iperparametri trovati (dal Fold {best_fold['fold']}): {best_hyperparameters}")

        logger.info("Addestramento del modello finale su tutto il set di train/validazione...")
        full_train_val_dataset = HARDataset(X, Y_mapped, all_window_indices, all_consecutivity)
        final_train_loader = DataLoader(full_train_val_dataset, batch_size=best_hyperparameters['batch_size'], shuffle=True, collate_fn=collate_fn)

        final_model = DeepConvLSTM()
        final_model.load_state_dict(torch.load(pretrained_model_path, map_location='cpu'), strict=False)
        final_model = configure_model_for_tuning(final_model, args.tuning_strategy, num_classes)

        #addestro modello finale
        final_f1_train, final_best_state= train_with_cm.train(net=final_model, 
            train_loader=final_train_loader, 
            val_loader=None, 
            exp_figures_dir=exp_figures_dir,
            exp_reports_dir=exp_reports_dir,
            epochs=args.epochs, 
            lr=best_hyperparameters['lr'],
            figure_name="final_training_on_all_train_val_data",
            save_plots=False
        )

        # carico e salvo il MIGLIOR modello emerso durante quel training
        final_model.load_state_dict(final_best_state, strict=False)
        final_model_path = os.path.join(exp_models_dir, "final_model.pkl")
        torch.save(final_model.state_dict(), final_model_path)
        logger.info(f"Modello finale salvato in: {final_model_path}")


        logger.info("Valutazione del modello finale su tutti i dati di training...")

        #creo data loader, uguale a final_train_loader con l'unica differenza che non faccio shuffle
        final_eval_loader = DataLoader(full_train_val_dataset, 
                                    batch_size=best_hyperparameters['batch_size'], 
                                    shuffle=False, 
                                    collate_fn=collate_fn)

        train_with_cm.evaluate_model(
            net=final_model,
            test_loader=final_eval_loader,
            exp_figures_dir=exp_figures_dir,
            exp_reports_dir=exp_reports_dir,
            figure_name="FINAL_training_evaluation", 
            save_confusion_matrix=True,
            save_predictions_csv=True,
            save_f1_score=True,
            labels_dict=labels_dict_mapped
        )

        # Crea il DataLoader per il test finale.
        test_dataset_final = HARDataset(X_test_final, Y_test_mapped_balanced, 
                                window_indices=indices_test_final, 
                                consecutivity=consec_test_final)
        test_loader_final = DataLoader(test_dataset_final, batch_size=best_hyperparameters['batch_size'], collate_fn=collate_fn)

        logger.info("Valutazione finale del modello sul hold-out test set...")
        train_with_cm.evaluate_model(
            net=final_model, 
            test_loader=test_loader_final,
            exp_figures_dir=exp_figures_dir,
            exp_reports_dir=exp_reports_dir,
            figure_name="FINAL_holdout_evaluation",
            save_confusion_matrix=True,
            save_predictions_csv=True,
            save_f1_score=True,
            labels_dict=labels_dict_mapped
        )
    else:
        logger.info("AVVIO PROCEDURA: K-Fold su tutti i dati per trovare HP + Training/Valutazione finale")

        logger.info("\n--- Fase 1: Esecuzione K-Fold per trovare i migliori iperparametri e performance media ---")
        # Inizializza StratifiedKFold.
        skf = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=seed_used)
        
        # Lista per conservare i risultati dettagliati di ogni fold.
        all_fold_scores = []


        # Dizionario per i risultati
        fold_results = {
            'fold': [], 'best_f1_score': [], 'best_params': [], 'test_f1_score': []
        }

        

        for fold, (train_val_idx, test_idx) in enumerate(skf.split(X, Y_mapped)):
            
            # --- BLOCCO DI DEBUG DEGLI INDICI ---
            logger.info(f"\n===== FOLD {fold + 1}/{args.k_folds} | DEBUG DELLO SPLIT =====")
            logger.info(f"Train+Val indices totali: {len(train_val_idx)}")
            logger.info(f"Test indices totali: {len(test_idx)}")
            logger.debug(f"Primi 10 Train+Val indices: {train_val_idx[:10]}")
            logger.debug(f"Primi 10 Test indices: {test_idx[:10]}")
            
            overlap = set(train_val_idx).intersection(set(test_idx))
            if overlap: logger.error(f"ERRORE: SOVRAPPOSIZIONE TROVATA TRA TRAIN E TEST: {overlap}")
            else: logger.info("Controllo sovrapposizione: OK. Nessuna sovrapposizione.")
            
            test_window_ids = [all_window_indices[i]['global_window_id'] for i in test_idx]
            logger.info(f"Global IDs delle finestre nel set di test (primi 10): {test_window_ids[:10]}")
        

            X_train_val, X_test_fold = X[train_val_idx], X[test_idx]
            Y_train_val, Y_test_fold = Y_mapped[train_val_idx], Y_mapped[test_idx]

            train_val_indices_meta = [all_window_indices[i] for i in train_val_idx]
            test_indices_meta = [all_window_indices[i] for i in test_idx]
            train_val_consec = all_consecutivity[train_val_idx]
            test_consec = all_consecutivity[test_idx]
            
            split_idx = int(0.8 * len(X_train_val))
            X_train_fold, Y_train_fold = X_train_val[:split_idx], Y_train_val[:split_idx]
            X_val_fold, Y_val_fold = X_train_val[split_idx:], Y_train_val[split_idx:]
            
            train_indices_meta = train_val_indices_meta[:split_idx]
            val_indices_meta = train_val_indices_meta[split_idx:]
            train_consec = train_val_consec[:split_idx]
            val_consec = train_val_consec[split_idx:]
            
            if args.ft_aug == 'alltrs':
                logger.info("Applicazione data augmentation al set di training del fold.")
                transformation_function = generate_composite_transform_function_simple([
                    noise_transform_vectorized, scaling_transform_vectorized, negate_transform_vectorized,
                    time_flip_transform_vectorized, channel_shuffle_transform_vectorized,
                    time_warp_transform_improved, time_warp_transform_low_cost
                ])
                X_train_aug = transformation_function(X_train_fold)
                X_train_fold = np.concatenate([X_train_fold, X_train_aug], axis=0)
                Y_train_fold = np.concatenate([Y_train_fold, Y_train_fold.copy()], axis=0)
                train_indices_meta.extend(train_indices_meta.copy())
                train_consec = np.concatenate([train_consec, train_consec.copy()], axis=0)

            train_dataset = HARDataset(X_train_fold, Y_train_fold, train_indices_meta, train_consec)
            val_dataset = HARDataset(X_val_fold, Y_val_fold, val_indices_meta, val_consec)
            test_dataset = HARDataset(X_test_fold, Y_test_fold, test_indices_meta, test_consec)

            logger.info(f"--- Inizio ottimizzazione iperparametri per il Fold {fold + 1} ---")
            def objective_fold(trial):
                #lr = trial.suggest_categorical('lr', [1e-5, 1e-4, 1e-3]) VALIDATION LOSS PIATTA 
                lr = trial.suggest_categorical('lr', [5e-4, 1e-3, 5e-3, 1e-2]) 
                
                batch_size = trial.suggest_categorical('batch_size', [2, 4])
                loss_type = trial.suggest_categorical('loss_type', [
                    'focal', 'label_smoothing', 'weighted_ce', 'combined'
                ])
                criterion = None
                if loss_type == 'focal':
                    # Per FocalLoss, suggerisce un valore per gamma.
                    gamma = trial.suggest_categorical('gamma', [1.0, 2.0, 3.0])
                    # (Assicurati che FocalLoss sia importata correttamente)
                    criterion = FocalLoss(gamma=gamma, num_classes=num_classes, task_type='multi-class')
                elif loss_type == 'label_smoothing':
                    # Per LabelSmoothing, suggerisce un valore di smoothing.
                    smoothing = trial.suggest_categorical('smoothing', [0.05, 0.1, 0.15, 0.2])
                    # (Assicurati che LabelSmoothingCrossEntropy sia importata)
                    criterion = LabelSmoothingCrossEntropy(epsilon=smoothing)
                    
                elif loss_type == 'combined':
                    # Per la loss combinata, suggerisce i parametri specifici.
                    gamma = trial.suggest_categorical('gamma_combined', [1.0, 2.0]) # Nome diverso per evitare conflitti
                    focal_weight = trial.suggest_categorical('focal_weight', [0.3, 0.5, 0.7])
                    criterion = CombinedLoss(
                        focal_weight=focal_weight,
                        ce_weight=1.0 - focal_weight,
                        gamma=gamma,
                        class_weights=class_weights_tensor, # Usa i pesi calcolati prima
                        num_classes=num_classes
                    )
                else:  # 'weighted_ce'
                    # Per la Cross Entropy pesata, usa i pesi calcolati prima.
                    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
                
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, collate_fn=collate_fn)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=collate_fn)
                
                model = DeepConvLSTM()
                model.load_state_dict(torch.load(pretrained_model_path, map_location='cpu'), strict=False)
                model = configure_model_for_tuning(model, args.tuning_strategy, num_classes)
                logger.info(f"TRIAL {trial.number}: lr={lr}, batch={batch_size}, loss={loss_type}")
                # Non salviamo i risultati per i trial intermedi di Optuna
                best_f1, best_state = train_with_cm.train( net=model, 
                    train_loader=train_loader, 
                    val_loader=val_loader,
                    exp_figures_dir=exp_figures_dir,
                    exp_reports_dir=exp_reports_dir,
                    epochs=args.epochs, 
                    lr=lr,
                    criterion=criterion,
                    save_plots=False 
                )

                trial.set_user_attr("best_state", best_state)
                return best_f1
            
            # Ottimizzazione per questo fold
            logger.info(f"Inizio ottimizzazione iperparametri per fold {fold + 1}")
            study = optuna.create_study(direction='maximize', pruner=MedianPruner())
            study.optimize(objective_fold, n_trials=args.n_trials)
            
            
            best_params_fold = study.best_params
            best_f1_fold = study.best_value
            best_trial = study.best_trial
            best_state = best_trial.user_attrs["best_state"]
            logger.info(f"Migliori iperparametri per il fold {fold + 1}: {best_params_fold} (Val F1: {best_f1_fold:.4f})")

            model_final_fold = DeepConvLSTM()
            model_final_fold.load_state_dict(torch.load(pretrained_model_path, map_location='cpu'), strict=False)
            model_final_fold = configure_model_for_tuning(model_final_fold, args.tuning_strategy, num_classes)
            model_final_fold.load_state_dict(best_state, strict=False) 

            test_loader = DataLoader(test_dataset, batch_size=best_params_fold['batch_size'], shuffle=False, drop_last=False, collate_fn=collate_fn)

            logger.info(f"\n=== DEBUG DATALOADER FOLD {fold + 1} ===")
            logger.info(f"Test dataset size: {len(test_dataset)}")
            

            # Conta quanti batch effettivi

            test_batches = len(test_loader)

            # CALCOLO CORRETTO delle finestre usate (con drop_last=False usa TUTTE)

            test_samples_used = len(test_dataset)

            logger.info(f"Test batches: {test_batches} (finestre usate: {test_samples_used}/{len(test_dataset)})")


            # --- BLOCCO DI DEBUG PRE-VALUTAZIONE ---
            logger.info(f"\n=== PRE-EVALUATE DEBUG (Fold {fold+1}) ===")
            logger.info(f"Test dataset size prima di evaluate: {len(test_dataset)}")
            logger.info(f"Test loader size prima di evaluate: {len(test_loader)}")
            if len(test_loader) > 0:
                for i, batch in enumerate(test_loader):
                    if i == 0:
                        inputs, targets, indices, _ = batch
                        logger.info(f"Primo batch di test - inputs shape: {inputs.shape}")
                        logger.info(f"Primo batch di test - targets shape: {targets.shape}")
                        logger.info(f"Primo batch di test - indices: {indices}")
                        break

            logger.info(f"Valutazione del modello del Fold {fold + 1} sul suo test set...")
            _, _, test_f1_fold = train_with_cm.evaluate_model(
                net=model_final_fold, 
                test_loader=test_loader,
                exp_figures_dir=exp_figures_dir,
                exp_reports_dir=exp_reports_dir,
                figure_name=f"TEST_fold_{fold + 1}", # Nome file base per salvare CM e predizioni
                save_confusion_matrix=True,
                save_predictions_csv=True,
                save_f1_score=True,
                labels_dict=labels_dict_mapped
            )

            # Salva il modello del fold.
            model_path_fold = os.path.join(exp_models_dir, f"model_fold_{fold + 1}.pkl")
            torch.save(model_final_fold.state_dict(), model_path_fold)

            fold_results['fold'].append(fold + 1)
            fold_results['best_f1_score'].append(best_f1_fold)
            fold_results['best_params'].append(best_params_fold)
            fold_results['test_f1_score'].append(test_f1_fold)
            
            all_fold_scores.append({
                'fold': fold + 1, 'params': best_params_fold, 'val_f1': best_f1_fold,
                'test_f1': test_f1_fold, 'model_path': model_path_fold
            })
            
            logger.info(f"Fold {fold + 1} completato - Test F1: {test_f1_fold:.4f}")

        logger.info("Aggregazione delle matrici di confusione di tutti i fold...")
        combine_kfold_confusion_matrices(
            exp_reports_dir=exp_reports_dir,
            num_folds=args.k_folds,
            class_names=[labels_dict_mapped[i] for i in range(num_classes)],
            exp_figures_dir=exp_figures_dir
        )

        # 2.2 Identifica i migliori iperparametri globali
        logger.info("Identificazione dei migliori iperparametri globali...")
        best_fold = max(all_fold_scores, key=lambda x: x['val_f1'])
        best_hyperparameters = best_fold['params']
        logger.info(f"Migliori iperparametri globali scelti (dal Fold {best_fold['fold']}): {best_hyperparameters}")

        # 2.3 Addestra un modello finale su TUTTI i dati (X e Y)
        logger.info("Addestramento del modello finale su tutti i dati disponibili...")
        final_dataset = HARDataset(X, Y_mapped, all_window_indices, all_consecutivity)
        final_train_loader = DataLoader(final_dataset, batch_size=best_hyperparameters['batch_size'], shuffle=True, collate_fn=collate_fn)
        
        final_model = DeepConvLSTM()
        final_model.load_state_dict(torch.load(pretrained_model_path, map_location='cpu'), strict=False)
        final_model = configure_model_for_tuning(final_model, args.tuning_strategy, num_classes)
        
        
        #addestro modello finale
        final_f1_train, final_best_state = train_with_cm.train(
            net=final_model,
            train_loader=final_train_loader,
            val_loader=None,  # no validation
            exp_figures_dir=exp_figures_dir,
            exp_reports_dir=exp_reports_dir,
            epochs=args.epochs,
            lr=best_hyperparameters['lr'],
            figure_name="final_training_on_all_train_val_data",
            save_plots=False
        )

        
       # carico e salvo il MIGLIOR modello emerso durante quel training
        final_model.load_state_dict(final_best_state, strict=False)
        final_model_path = os.path.join(exp_models_dir, "final_model.pkl")
        torch.save(final_model.state_dict(), final_model_path)
        logger.info(f"Modello finale salvato in: {final_model_path}")
        
        logger.info("Valutazione del modello finale su tutti i dati di training...")

        #creo data loader, uguale a final_train_loader con l'unica differenza che non faccio shuffle
        final_eval_loader = DataLoader(final_dataset, 
                                    batch_size=best_hyperparameters['batch_size'], 
                                    shuffle=False, 
                                    collate_fn=collate_fn)

        train_with_cm.evaluate_model(
            net=final_model,
            test_loader=final_eval_loader,
            exp_figures_dir=exp_figures_dir,
            exp_reports_dir=exp_reports_dir,
            figure_name="FINAL_training_evaluation", 
            save_confusion_matrix=True,
            save_predictions_csv=True,
            save_f1_score=True,
            labels_dict=labels_dict_mapped
        )


    
if __name__ == "__main__":
    args = parse_args()
    main(args)