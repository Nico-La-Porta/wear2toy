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
    parser.add_argument('--pt-norm', type=str, default='none', choices=['none', 'meanstd', 'iqr'],
                        help='Normalizzazione usata durante il pre-training per selezionare il modello corretto.')
    parser.add_argument('--pt-aug', type=str, default='none', choices=['none', 'alltrs'],
                        help='Data augmentation usata durante il pre-training per selezionare il modello corretto.')

    # Argomenti per il Fine-Tuning (FT)
    parser.add_argument('--ft-norm', type=str, default='none', choices=['none', 'meanstd', 'iqr'],
                        help='Metodo di normalizzazione per il fine-tuning. "meanstdonpt" usa le statistiche del pre-training.')
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

        
def main(args):
    """
    Funzione principale che esegue l'esperimento.
    """

    data_path = "C:\\codes\\HumanActivityRecognition\\data\\downstream_data"
    #data_path = DOWNSTREAM_DATA_DIR



    # ===== Ricava il seed dal pickle =====
    target_actions = None
    seed_used = 42
    npz_path = None
    if args.filtered_indices_pkl:
        try:
            with open(args.filtered_indices_pkl, 'rb') as f:
                idx_pack = pickle.load(f)
            target_actions = idx_pack.get('target_actions', None)
            seed_used = idx_pack.get('processing_info', {}).get('seed', None)

            # Costruisci il path del file NPZ basandoti sul pickle
            pkl_dir = os.path.dirname(args.filtered_indices_pkl)
            npz_filename = f'{args.toy}_processed_windows_seed{seed_used}.npz'
            npz_path = os.path.join(pkl_dir, npz_filename)


            if seed_used is None:
                # fallback: estrai dal nome file se contiene "seed_xx"
                import re
                match = re.search(r'seed[_\-]?(\d+)', args.filtered_indices_pkl)
                if match:
                    seed_used = int(match.group(1))
                    npz_filename = f'{args.toy}_processed_windows_seed{seed_used}.npz'
                    npz_path = os.path.join(pkl_dir, npz_filename)
        except Exception as e:
            logger.warning(f"Impossibile leggere seed dal pickle: {e}")
    if seed_used is None:
        seed_used = 42  # fallback finale

    if npz_path is None or not os.path.exists(npz_path):
        raise FileNotFoundError(f"File NPZ non trovato: {npz_path}. Assicurati di aver eseguito zupt_analysis_2.py prima.")

    

    logger.info(f"Caricamento finestre pre-processate da: {npz_path}")
    logger.info(f"Target actions: {target_actions}")
    logger.info(f"Seed utilizzato: {seed_used}")


    # ===== 2. CARICA DATI ORIGINALI PER CALCOLO MEDIA/STD =====
    logger.info("Caricamento dati originali per calcolo statistiche di normalizzazione...")
    df_toy, toy_mapping = load_and_preprocess_data_unified(
    toy_name=args.toy, 
    data_path=data_path,
    target_actions=target_actions 
    )
    
    df_toy = add_consecutive_segment_id(df_toy)
    logger.info(f"Dati originali caricati: {df_toy.shape}")
    logger.info(f"Classi presenti: {sorted(df_toy['action_id'].unique())}")




    # ===== 3. CARICA FINESTRE PRE-PROCESSATE DA NPZ =====
    logger.info("Caricamento finestre pre-processate da NPZ...")
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


    
    

    
    # ===== 4. SETUP RIPRODUCIBILITÀ =====
    np.random.seed(seed_used)
    random.seed(seed_used)
    torch.manual_seed(seed_used)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_used)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



    # ===== 5. SETUP DIRECTORIES =====
    exp_name = (
        f"T{args.toy}_PTN{args.pt_norm}_PTA{args.pt_aug}_"
        f"FTN{args.ft_norm}_FTA{args.ft_aug}_TS{args.tuning_strategy}"
    )


    exp_root_models  = os.path.join(MODELS_DIR,  exp_name)
    exp_root_figures = os.path.join(FIGURES_DIR, exp_name)
    exp_root_reports = os.path.join(REPORTS_DIR, exp_name)
    os.makedirs(exp_root_models, exist_ok=True)
    os.makedirs(exp_root_figures, exist_ok=True)
    os.makedirs(exp_root_reports, exist_ok=True)


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

    logger.info(f"Finestre caricate da NPZ")
    

    # ===== 6. CALCOLO STATISTICHE DI NORMALIZZAZIONE =====
    mean, std = None, None
    median, iqr = None, None
    
    if args.ft_norm == 'meanstd':
        logger.info("Normalizzazione con statistiche del pre-training (MEAN).")
        mean_df, std_df = pd.read_csv(os.path.join(REPORTS_DIR, 'mean_trs.csv')), pd.read_csv(os.path.join(REPORTS_DIR, 'std_trs.csv'))
        mean, std = mean_df['mean'].values[1:-1], std_df['std'].values[1:-1]
        logger.info("Parametri di normalizzazione caricati dal pre-training")
        logger.debug(f"Mean shape: {mean.shape}, Std shape: {std.shape}")
    elif args.ft_norm == 'iqr':
        logger.info("Normalizzazione con statistiche del pre-training (IQR).")
        median_df, iqr_df = pd.read_csv(os.path.join(REPORTS_DIR, 'median_trs.csv')), pd.read_csv(os.path.join(REPORTS_DIR, 'iqr_trs.csv'))
        median, iqr = median_df['median'].values[1:-1], iqr_df['iqr'].values[1:-1]
        logger.info("Parametri di normalizzazione caricati dal pre-training")
        logger.debug(f"Mean shape: {median.shape}, Std shape: {iqr.shape}")
    

    # ===== 7. APPLICAZIONE NORMALIZZAZIONE ALLE FINESTRE =====
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
    elif median is not None and iqr is not None:
        logger.info("Applicazione normalizzazione IQR alle finestre pre-processate...")
        
        # Normalizza tutte le finestre
        original_shape = X.shape
        X_reshaped = X.reshape(-1, X.shape[-1])  # (n_windows * timesteps, n_features)
        
        # Applica normalizzazione IQR
        X_normalized = (X_reshaped - median) / iqr
        X = X_normalized.reshape(original_shape)  # Ripristina forma originale
        
        logger.info(f"Normalizzazione IQR applicata a tutte le finestre")
        logger.info(f"Shape dopo normalizzazione: {X.shape}")
    else:
        logger.info("Nessuna normalizzazione applicata alle finestre")
    

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


        
        # ===== 10. VERIFICA NECESSITÀ BILANCIAMENTO =====
        train_actions_set = set(Y_train_val)
        test_actions_set = set(Y_test_holdout)
        missing_in_test = train_actions_set - test_actions_set

        if missing_in_test:
            (X_train_val, Y_train_val, kid_ids_train_val, action_ids_train_val,
            original_files_train_val, row_start_train_val, row_end_train_val, padding_train_val,
            X_test_holdout, Y_test_holdout, kid_ids_test_holdout, action_ids_test_holdout,
            original_files_test_holdout, row_start_test_holdout, row_end_test_holdout,
            padding_test_holdout) = balance_holdout_classes_npz(
                X_train_val, Y_train_val, kid_ids_train_val, action_ids_train_val,
                X_test_holdout, Y_test_holdout, kid_ids_test_holdout, action_ids_test_holdout,
                original_files_train_val, original_files_test_holdout,
                row_start_train_val, row_start_test_holdout,
                row_end_train_val, row_end_test_holdout,
                padding_train_val, padding_test_holdout,
                percentage_to_move=0.15, random_seed=seed_used
            )

            
            logger.info("Bilanciamento applicato con successo!")
        else:
            logger.info("Nessun bilanciamento necessario: tutte le classi presenti nel test")

        # ===== 9. MAPPATURA AZIONI =====
        logger.info("Mappatura delle azioni per la classificazione...")
        
        # Usa solo le azioni del training per creare la mappatura
        unique_labels = np.unique(Y_train_val)
        label_mapping = {label: i for i, label in enumerate(unique_labels)}
        Y_train_val_mapped = np.array([label_mapping[y] for y in Y_train_val])
        Y_test_holdout_mapped = np.array([label_mapping[y] for y in Y_test_holdout])
        num_classes = len(unique_labels)

        
        logger.info(f"Azioni originali: {sorted(unique_labels)}")
        logger.info(f"Mappatura azioni: {label_mapping}")
        logger.info(f"Numero classi finali: {num_classes}")


        # ===== 11. RICOSTRUISCI METADATI PER COMPATIBILITÀ =====
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

    # Mappa anche le etichette del test holdout se presente
    if args.holdout_kids:
        Y_test_holdout_mapped = np.array([label_mapping[y] for y in Y_test_holdout])
        logger.info(f"Test holdout mappato: {len(Y_test_holdout_mapped)} campioni")
    
    # ===== 10. CALCOLO PESI DELLE CLASSI =====

    # Calcola i pesi in base alla frequenza inversa delle classi.
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(Y_mapped),
        y=Y_mapped
    )
    # Converte i pesi in un tensore PyTorch, pronto per essere usato dalla loss.
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
    if torch.cuda.is_available():
        class_weights_tensor = class_weights_tensor.to('cuda')
    
    logger.info(f"Pesi delle classi calcolati: {dict(zip(np.unique(Y_mapped), class_weights))}")

    # ===== 11. DISTRIBUZIONE INIZIALE =====
    # Converti Y_mapped di nuovo alle azioni originali per la visualizzazione
    inv_label_mapping = {v: k for k, v in label_mapping.items()}
    Y_original_for_plot = np.array([inv_label_mapping[y] for y in Y_mapped])
    
    log_and_plot_distribution(
        Y=Y_original_for_plot,
        title_prefix="Distribuzione Finestre da NPZ",
        toy_name=args.toy,
        toy_mapping=toy_mapping,
        save_dir=exp_figures_dir
    )

    # ===== 12. SETUP MODELLO PRE-ADDESTRATO =====
    pretrained_model_path = get_pretrained_model_path(args.pt_norm, args.pt_aug)
    logger.info(f"Utilizzo del modello pre-addestrato: {pretrained_model_path}")


    # ===== 13. CREA METADATI FINESTRE PER COMPATIBILITÀ =====
    # Ricostruisci i metadati nel formato atteso dal resto del codice
    all_window_indices = []
    all_consecutivity = np.ones(len(X), dtype=bool)  # Assume tutte consecutive per ora
    
    for i in range(len(X)):
        all_window_indices.append({
            'global_window_id': i,
            'action_id': int(action_ids[i]),
            'kid_id': str(kid_ids[i]),
            'original_file': str(original_files[i]),
            'row_indices': list(range(int(row_indices_start[i]), int(row_indices_end[i]) + 1)) if row_indices_start[i] != -1 else [],
            'padding_count': int(padding_counts[i])
        })

    logger.info(f"Metadati finestre ricostruiti: {len(all_window_indices)} finestre")
    # ===== 14. PROCEDURE PRINCIPALI =====
    logger.info(f"Dati pronti per l'esperimento: X.shape={X.shape}, num_classes={num_classes}")

    if args.holdout_kids:
        logger.info("AVVIO PROCEDURA: K-Fold su Train/Val set + Valutazione finale su Hold-out")

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
            
            # Split train/validation interno
            split_idx = int(0.8 * len(X_train_val))
            X_train_fold, Y_train_fold = X_train_val[:split_idx], Y_train_val[:split_idx]
            X_val_fold, Y_val_fold = X_train_val[split_idx:], Y_train_val[split_idx:]
            
            train_indices_meta = train_val_indices_meta[:split_idx]
            val_indices_meta = train_val_indices_meta[split_idx:]
            train_consec = train_val_consec[:split_idx]
            val_consec = train_val_consec[split_idx:]


            # Data augmentation se richiesta
            if args.ft_aug == 'alltrs':
                logger.info("Applicazione data augmentation al set di training del fold.")
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
        # Salva i risultati dei fold in un JSON file
        fold_results_path = os.path.join(exp_reports_dir, "best_hyperparameters.json")
        with open(fold_results_path, 'w') as f:
            json.dump(fold_results, f, indent=4)

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

        # Ricostruisci metadati per tutto il dataset
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


        skf = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=seed_used)
        all_fold_scores = []
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
                    time_flip_transform_vectorized, intra_sensor_channel_shuffle_transform_vectorized,
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
            # Salva i migliori iperparametri in un JSON file
            with open(os.path.join(exp_reports_dir, f"best_hyperparameters_fold_{fold + 1}.json"), 'w') as f:
                json.dump(best_params_fold, f, indent=4)

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
        # Salva i risultati dei fold in un JSON file
        fold_results_path = os.path.join(exp_reports_dir, "best_hyperparameters.json")
        with open(fold_results_path, 'w') as f:
            json.dump(fold_results, f, indent=4)

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