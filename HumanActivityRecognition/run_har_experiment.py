import os
import sys
import argparse
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import torch
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
from utils.focal_loss import FocalLoss, LabelSmoothingCrossEntropy, WeightedCrossEntropyLoss, CombinedLoss
from utils.log_config import logger
from utils.figures import combine_kfold_confusion_matrices
from utils import mapping_activity
from utils.transformations import *
from utils.transformations_utils import *



# FUNZIONE DI DOWNSAMPLING SELETTIVO
# =============================================================================
logger.info("=== DEFINIZIONE FUNZIONE DI DOWNSAMPLING ===")

def count_padding_in_window(window):
    """
    Conta il numero di time steps (righe) che sono interamente padding (tutti zeri).
    """
    if window.ndim == 1:
        # Se la finestra è 1D, conta i singoli zeri
        return np.sum(window == 0)
    else:
        # Se la finestra è 2D (time_steps, features), conta le righe di soli zeri
        padding_rows = np.all(window == 0, axis=1)
        return np.sum(padding_rows)

def selective_downsampling_by_padding(X, Y, kid_action_counts, toy_name, max_windows_per_kid_action=50, toys_to_downsample=None):
    """
    Filtra le finestre per specifiche categorie di giocattoli, limitandole a 
    max_windows_per_kid_action finestre per ogni combinazione bambino/azione,
    eliminando prima quelle con più padding.
    
    Args:
        X (np.array): L'array delle finestre dei dati.
        Y (np.array): L'array delle etichette corrispondenti.
        kid_action_counts (dict): Dizionario con la struttura {action_key: {kid_id: count}}.
        max_windows_per_kid_action (int): Numero massimo di finestre da mantenere per gruppo.
        toys_to_downsample (list): Lista di stringhe dei nomi dei giocattoli da sottoporre a downsampling 
                                     (es. ['car', 'doll', 'elephant']). Se None, non viene applicato alcun filtro.
    
    Returns:
        tuple: (X_filtered, Y_filtered, kid_action_counts_filtered)
    """
    if toys_to_downsample is None:
        logger.warning("Nessun giocattolo specificato per il downsampling. Restituzione dei dati originali.")
        return X, Y, kid_action_counts

    logger.info(f"Avvio downsampling selettivo per i giocattoli: {toys_to_downsample}")
    logger.info(f"Numero massimo di finestre per bambino/azione target: {max_windows_per_kid_action}")

    # Mappatura di ogni finestra con i suoi metadati (indice, bambino, azione, padding)
    window_metadata = []
    current_idx = 0
    for action_key, kid_dict in kid_action_counts.items():
        
        action_name = str(Y[current_idx]) #Ricava l'etichetta dall'array Y all’indice corrente
        
        for kid_id, window_count in kid_dict.items():
            for i in range(window_count):
                idx = current_idx + i
                window_metadata.append({
                    'index': idx,
                    'kid_id': kid_id,
                    'action_key': action_key, # Mantengo chiave originale
                    'padding_count': count_padding_in_window(X[idx])
                })
            current_idx += window_count
    
    # Raggruppamento delle finestre per bambino e azione
    kid_action_groups = defaultdict(list)
    for metadata in window_metadata:
        key = (metadata['kid_id'], metadata['action_key'])
        kid_action_groups[key].append(metadata)

    # Logica di filtraggio
    indices_to_keep = []
    
    for (kid_id, action_key), windows in kid_action_groups.items():
        # Decide se applicare il downsampling: vero se qualunque stringa in toys_to_downsample è un sottostringa di action_key
        # (Esempio: se action_key="car_push" e toys_to_downsample=["car"] → True.)
        apply_downsampling = any(toy in action_key for toy in toys_to_downsample)

        if apply_downsampling and len(windows) > max_windows_per_kid_action:
            # Ordina le finestre in base al padding (crescente, quindi meno padding prima)
            windows_sorted = sorted(windows, key=lambda w: w['padding_count'])
            
            # Seleziona le migliori 'max_windows_per_kid_action' finestre
            selected_windows = windows_sorted[:max_windows_per_kid_action]
            indices_to_keep.extend([w['index'] for w in selected_windows])
            
            logger.info(f"Gruppo '{kid_id}-{action_key}': ridotto da {len(windows)} a {len(selected_windows)} finestre.")
        else:
            # Mantieni tutte le finestre se non superano il limite o se il giocattolo non è target
            indices_to_keep.extend([w['index'] for w in windows])
            if apply_downsampling:
                 logger.info(f"Gruppo '{kid_id}-{action_key}': mantenute tutte le {len(windows)} finestre (sotto la soglia).")
            # else:
            #      logger.info(f"Gruppo '{kid_id}-{action_key}': mantenute tutte le {len(windows)} finestre (giocattolo non target).")


    # Creazione dei nuovi array e del dizionario dei conteggi
    indices_to_keep.sort()
    X_filtered = X[indices_to_keep]
    Y_filtered = Y[indices_to_keep]

    # Ricostruisco kid_action_counts_filtered basandomi sui dati filtrati
    kid_action_counts_filtered = defaultdict(lambda: defaultdict(int))
    # È necessario un modo per mappare gli indici filtrati a kid_id e action_key: utilizzo i metadati, ma solo per gli indici che abbiamo mantenuto
    kept_metadata = [meta for meta in window_metadata if meta['index'] in indices_to_keep]
    for meta in kept_metadata:
        kid_action_counts_filtered[meta['action_key']][meta['kid_id']] += 1

    logger.info("Downsampling completato.")
    logger.info(f"Finestre originali: {len(X)}")
    logger.info(f"Finestre mantenute: {len(X_filtered)}")
    logger.info(f"Finestre rimosse: {len(X) - len(X_filtered)}")
    
    return X_filtered, Y_filtered, dict(kid_action_counts_filtered)
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
    parser.add_argument('--random-state', type=int, default=42, help='Seed per la riproducibilità.')

    return parser.parse_args()




def get_pretrained_model_path(pt_norm, pt_aug):
    """
    Restituisce il path del modello pre-addestrato corretto in base alla configurazione.
    """
    base_path = r'D:\codes\HumanActivityRecognition\models'
    
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


def load_and_preprocess_data(toy_name, data_path):
    """
    Carica e pre-elabora i dati per un giocattolo specifico.
    """
    logger.info(f"Caricamento e pre-elaborazione per il giocattolo: {toy_name}")
    
    toy_configs = {
        'ball': {'prefix': 'BA', 'mapping': mapping_activity.BALL_ACTION_MAPPING, 'classes_to_remove': []},
        'car': {'prefix': 'C', 'mapping': mapping_activity.CAR_ACTION_MAPPING, 'classes_to_remove': [2, 3, 5, 9, 11, 12, 14, 16, 18, 19, 27, 28, 29, 31, 32, 37, 38, 39, 40]},
        'doll': {'prefix': 'DO', 'mapping': mapping_activity.DOLL_ACTION_MAPPING, 'classes_to_remove': [3, 4, 7, 9, 12, 19, 27, 31]},
        'spoon': {'prefix': 'SP', 'mapping': mapping_activity.SPOON_ACTION_MAPPING, 'classes_to_remove': [4]},
        'elephant': {'prefixes': ["BE", "GE", "RE", "YE", "WE"], 'mapping': mapping_activity.ELEPHANT_ACTION_MAPPING, 'classes_to_remove': [7, 20, 5, 2, 19, 29, 31]},
    }
    
    config = toy_configs[toy_name]
    final_csv_path = os.path.join(data_path, f'df_{config["prefix"] if "prefix" in config else "ELEP"}_non_null.csv')

    if os.path.exists(final_csv_path):
        df_toy = pd.read_csv(final_csv_path)
    else:
        files = []
        if 'prefixes' in config:
            for p in config['prefixes']:
                files.extend(glob.glob(os.path.join(data_path, f"*_{p}*.csv")))
        else:
            files = glob.glob(os.path.join(data_path, f"*_{config['prefix']}*.csv"))
            
        df_list = [pd.read_csv(file) for file in files]
        df_toy = pd.concat(df_list)
        df_toy = df_toy[df_toy['action_id'] != 0]
        df_toy.to_csv(final_csv_path, index=False)

    if config['classes_to_remove']:
        df_toy = df_toy[~df_toy['action_id'].isin(config['classes_to_remove'])]
        

    return df_toy, config['mapping']



def main(args):
    """
    Funzione principale che esegue l'esperimento.
    """
    exp_name = (
        f"T{args.toy}_PTN{args.pt_norm}_PTA{args.pt_aug}_"
        f"FTN{args.ft_norm}_FTA{args.ft_aug}_TS{args.tuning_strategy}"
    )
    # Crea cartelle specifiche per questo esperimento
    exp_models_dir = os.path.join(MODELS_DIR, exp_name)
    exp_figures_dir = os.path.join(FIGURES_DIR, exp_name)
    exp_reports_dir = os.path.join(REPORTS_DIR, exp_name)
    os.makedirs(exp_models_dir, exist_ok=True)
    os.makedirs(exp_figures_dir, exist_ok=True)
    os.makedirs(exp_reports_dir, exist_ok=True)
    logger.info(f"===== INIZIO ESPERIMENTO: {exp_name} =====")

    data_path = "C:\\codes\\HumanActivityRecognition\\data\\downstream_data"
    df_toy, toy_mapping = load_and_preprocess_data(args.toy, data_path)
    logger.info(f"Classi presenti dopo il filtraggio: {df_toy['action_id'].unique()}")

    # --- Hold-out set per bambini specifici --- 
    df_train_val = None
    df_test_holdout = None
    if args.holdout_kids:
        logger.info(f"Separazione dei bambini per il test hold-out: {args.holdout_kids}")
        df_test_holdout = df_toy[df_toy['kid_id'].isin(args.holdout_kids)]
        df_train_val = df_toy[~df_toy['kid_id'].isin(args.holdout_kids)]
        logger.info(f"Dimensioni Training/Validation set: {df_train_val.shape}")
        logger.info(f"Dimensioni Hold-out Test set: {df_test_holdout.shape}")
    else:
        logger.info("Nessun hold-out set specificato: Esecuzione K-Fold su tutto il dataset.")
        df_train_val = df_toy.copy()

    # --- Normalizzazione ---
    df_normalized = pd.DataFrame()
    mean, std = None, None
    if args.ft_norm == 'onpt':
        logger.info("Normalizzazione con statistiche del pre-training.")
        mean_df, std_df = pd.read_csv(os.path.join(REPORTS_DIR, 'mean_trs.csv')), pd.read_csv(os.path.join(REPORTS_DIR, 'std_trs.csv'))
        mean, std = mean_df['mean'].values[1:-1], std_df['std'].values[1:-1]
        logger.info("Parametri di normalizzazione caricati dal pre-training")
        logger.debug(f"Mean shape: {mean.shape}, Std shape: {std.shape}")
    elif args.ft_norm == 'onft':
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

    # --- Salvataggio file per azione e Sliding Window ---
    logger.info("Salvataggio file per azione e applicazione sliding window...")
    X_list, Y_list, all_window_indices, all_consecutivity, kid_action_counts = [], [], [], [], defaultdict(dict)
    global_window_id = 0
    
    temp_action_dir = os.path.join(data_path, "temp_actions")
    os.makedirs(temp_action_dir, exist_ok=True)

    for action_id in df_normalized['action_id'].unique():
        df_action = df_normalized[df_normalized['action_id'] == action_id]
        temp_path = os.path.join(temp_action_dir, f'df_{args.toy}_action_{action_id}.csv')
        df_action.to_csv(temp_path, index=False)
        logger.debug(f"Salvato file temporaneo: {temp_path}")

        X_w, Y_w, kid_dict, is_consecutive, win_indices = sliding_window_on_data.process_csv(
            temp_path, 9, 100, 50
        )
        X_list.append(X_w); Y_list.append(Y_w); all_consecutivity.extend(is_consecutive)

        logger.info(f"Numero totale di finestre per l'azione {action_id}: {len(X_w)}")
        logger.info(f"Finestre consecutive per l'azione {action_id}: {sum(is_consecutive)}")
        logger.info(f"Finestre non consecutive per l'azione {action_id}: {len(is_consecutive) - sum(is_consecutive)}")
        logger.info(f"Numero  totale di finestre per l'azione {action_id}:{len(X_w)}")
        kid_action_counts[f"{args.toy}_action_{action_id}"] = kid_dict #dizionario principale per tenere traccia del numero di finestre per ogni bambino per ogni azione, ogni spoon_action è una chiave e il valore è un dizionario con il numero di finestre per ogni bambino
        logger.info(f"Contenuto finale di kid_action_counts: {kid_action_counts}") #per vedere quante finestre per ogni azione e per ogni bambino sono state elaborte 
        for i in range(len(X_w)):
            all_window_indices.append({'global_window_id': global_window_id, 'action_id': action_id, 'row_indices': win_indices[i].tolist()})
            global_window_id += 1
            
    
    shutil.rmtree(temp_action_dir) # Pulisce la cartella temporanea

    X = np.concatenate(X_list, axis=0)
    Y = np.concatenate(Y_list, axis=0).flatten()
    all_consecutivity = np.array(all_consecutivity)

    log_and_plot_distribution(
    Y=Y,
    title_prefix="Distribuzione Iniziale (Prima del Downsampling)",
    toy_name=args.toy,
    toy_mapping=toy_mapping,
    save_dir=exp_figures_dir
)

    # =============================================================================
    #  APPLICAZIONE DEL DOWNSAMPLING SELETTIVO
    # =============================================================================
    logger.info("=== CONTROLLO ED ESECUZIONE DOWNSAMPLING SELETTIVO ===")
    
    # Definisci i giocattoli su cui applicare il downsampling
    TARGET_TOYS_FOR_DOWNSAMPLING = ['car', 'doll', 'elephant']
    MAX_WINDOWS_PER_KID_ACTION = 50
    #  --- TEST PER VEDERE SE GLI INDICI COINCIDONO (da commentare una volta che ce ne siamo accertati) ---
    current_idx = 0
    for action_key, kid_dict in kid_action_counts.items():
        for kid_id, window_count in kid_dict.items():
            labels = Y[current_idx: current_idx + window_count]
            
            if not all(str(lab) in action_key for lab in labels):
                print(f"Mismatch trovato per {action_key}-{kid_id}")
            
            current_idx += window_count

    print("Controllo completato")
    # Applica la funzione solo se il giocattolo corrente è nella lista target
    # La funzione stessa si occuperà di tutto e aggiornerà le variabili X, Y e kid_action_counts
    X, Y, kid_action_counts = selective_downsampling_by_padding(
        X, Y, kid_action_counts, 
        toy_name=args.toy,
        max_windows_per_kid_action=MAX_WINDOWS_PER_KID_ACTION,
        toys_to_downsample=TARGET_TOYS_FOR_DOWNSAMPLING
    )
    # =============================================================================

    log_and_plot_distribution(
    Y=Y, 
    title_prefix="Distribuzione Finale (Dopo il Downsampling)",
    toy_name=args.toy,
    toy_mapping=toy_mapping,
    save_dir=exp_figures_dir
)


    unique_labels = np.unique(Y)
    label_mapping = {label: i for i, label in enumerate(unique_labels)}
    Y_mapped = np.array([label_mapping[y] for y in Y])
    num_classes = len(unique_labels)
    
    labels_dict = toy_mapping["encoded_to_name"]

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
        
        skf = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=args.random_state)
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
                best_f1 = train_with_cm.train( net=model, 
                    train_loader=train_loader, 
                    val_loader=val_loader,
                    exp_figures_dir=exp_figures_dir,
                    exp_reports_dir=exp_reports_dir,
                    epochs=args.epochs, 
                    lr=lr,
                    criterion=criterion,
                    save_plots=False 
                )
                return best_f1
            # Ottimizzazione per questo fold
            logger.info(f"Inizio ottimizzazione iperparametri per fold {fold + 1}")
            study = optuna.create_study(direction='maximize', pruner=MedianPruner())
            study.optimize(objective_fold, n_trials=args.n_trials)
            
            best_params_fold = study.best_params
            best_f1_fold = study.best_value
            logger.info(f"Migliori iperparametri per il fold {fold + 1}: {best_params_fold} (Val F1: {best_f1_fold:.4f})")

            model_final_fold = DeepConvLSTM()
            model_final_fold.load_state_dict(torch.load(pretrained_model_path, map_location='cpu'), strict=False)
            model_final_fold = configure_model_for_tuning(model_final_fold, args.tuning_strategy, num_classes)
            
            train_loader_final = DataLoader(train_dataset, batch_size=best_params_fold['batch_size'], shuffle=True, drop_last=False, collate_fn=collate_fn)
            val_loader_final = DataLoader(val_dataset, batch_size=best_params_fold['batch_size'], shuffle=False, drop_last=False, collate_fn=collate_fn)
            test_loader = DataLoader(test_dataset, batch_size=best_params_fold['batch_size'], shuffle=False, drop_last=False, collate_fn=collate_fn)

            logger.info(f"\n=== DEBUG DATALOADER FOLD {fold + 1} ===")
            logger.info(f"Train dataset size: {len(train_dataset)}")
            logger.info(f"Val dataset size: {len(val_dataset)}")
            logger.info(f"Test dataset size: {len(test_dataset)}")

            logger.info(f"\n=== DEBUG DATALOADER FOLD {fold + 1} ===")
            logger.info(f"Train dataset size: {len(train_dataset)}")
            logger.info(f"Val dataset size: {len(val_dataset)}")
            logger.info(f"Test dataset size: {len(test_dataset)}")
            

            # Conta quanti batch effettivi
            train_batches = len(train_loader_final)
            val_batches = len(val_loader_final)
            test_batches = len(test_loader)

            # CALCOLO CORRETTO delle finestre usate (con drop_last=False usa TUTTE)
            train_samples_used = len(train_dataset)
            val_samples_used = len(val_dataset)
            test_samples_used = len(test_dataset)

            logger.info(f"Train batches: {train_batches} (finestre usate: {train_samples_used}/{len(train_dataset)})")
            logger.info(f"Val batches: {val_batches} (finestre usate: {val_samples_used}/{len(val_dataset)})")
            logger.info(f"Test batches: {test_batches} (finestre usate: {test_samples_used}/{len(test_dataset)})")

            logger.info("Inizio training finale per il fold con i migliori iperparametri...")
            train_with_cm.train(net=model_final_fold, 
                train_loader=train_loader_final, 
                val_loader=val_loader_final,
                exp_figures_dir=exp_figures_dir,
                exp_reports_dir=exp_reports_dir,
                epochs=args.epochs, 
                lr=best_params_fold['lr'],
                figure_name=f"training_fold_{fold+1}", 
                save_plots=True
            )

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
                labels_dict=labels_dict,
                figures_dir=exp_figures_dir, # Specifica dove salvare i file
                reports_dir=exp_reports_dir  # Specifica dove salvare i report
            )

            model_path_fold = os.path.join(exp_models_dir, f"best_model_fold_{fold + 1}.pkl")
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
        train_with_cm.train(net=final_model, 
            train_loader=final_train_loader, 
            val_loader=None, 
            exp_figures_dir=exp_figures_dir,
            exp_reports_dir=exp_reports_dir,
            epochs=args.epochs, 
            lr=best_hyperparameters['lr'],
            figure_name="final_training_on_all_train_val_data",
            save_plots=False
        )

        # Salvo il modello finale addestrato, che è ora pronto per la valutazione.
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
            labels_dict=labels_dict
        )

        logger.info("Preparazione del hold-out test set per la valutazione finale...")

        # Il DataFrame df_test_holdout_normalized è già stato normalizzato correttamente all'inizio.
        # Ora applichiamo la sliding window su di esso.
        X_test_list, Y_test_list = [], []
        temp_action_dir_test = os.path.join(data_path, "temp_actions_test")
        os.makedirs(temp_action_dir_test, exist_ok=True)

        logger.info("Applicazione Sliding Window sul set di hold-out test...")
        for action_id in df_test_holdout_normalized['action_id'].unique():
            df_action_test = df_test_holdout_normalized[df_test_holdout_normalized['action_id'] == action_id]
            temp_path_test = os.path.join(temp_action_dir_test, f'df_test_{args.toy}_action_{action_id}.csv')
            df_action_test.to_csv(temp_path_test, index=False)
            
            # Chiamiamo la stessa funzione di sliding window
            X_w_test, Y_w_test, _, _, _ = sliding_window_on_data.process_csv(temp_path_test, 9, 100, 50)
            
            X_test_list.append(X_w_test)
            Y_test_list.append(Y_w_test)
            
        shutil.rmtree(temp_action_dir_test) # Pulisce la cartella temporanea
        
        # Concatena i risultati per creare gli array finali del test set.
        X_test_final = np.concatenate(X_test_list, axis=0)
        Y_test_final = np.concatenate(Y_test_list, axis=0).flatten()
        logger.info(f"Dati di test finali pronti: X_test_final.shape={X_test_final.shape}")

        # Applica lo STESSO remapping di etichette usato per il training.
        # È FONDAMENTALE usare lo stesso `label_mapping` per garantire coerenza.
        Y_test_final_mapped = np.array([label_mapping[y] for y in Y_test_final])

        # Crea il DataLoader per il test finale.
        test_dataset_final = HARDataset(X_test_final, Y_test_final_mapped)
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
            labels_dict=labels_dict
        )
    else:
        logger.info("AVVIO PROCEDURA: K-Fold su tutti i dati per trovare HP + Training/Valutazione finale")

        logger.info("\n--- Fase 1: Esecuzione K-Fold per trovare i migliori iperparametri e performance media ---")
        # Inizializza StratifiedKFold.
        skf = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=args.random_state)
        
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
                best_f1 = train_with_cm.train( net=model, 
                    train_loader=train_loader, 
                    val_loader=val_loader,
                    exp_figures_dir=exp_figures_dir,
                    exp_reports_dir=exp_reports_dir,
                    epochs=args.epochs, 
                    lr=lr,
                    criterion=criterion,
                    save_plots=False 
                )
                return best_f1
            # Ottimizzazione per questo fold
            logger.info(f"Inizio ottimizzazione iperparametri per fold {fold + 1}")
            study = optuna.create_study(direction='maximize', pruner=MedianPruner())
            study.optimize(objective_fold, n_trials=args.n_trials)
            
            best_params_fold = study.best_params
            best_f1_fold = study.best_value
            logger.info(f"Migliori iperparametri per il fold {fold + 1}: {best_params_fold} (Val F1: {best_f1_fold:.4f})")

            model_final_fold = DeepConvLSTM()
            model_final_fold.load_state_dict(torch.load(pretrained_model_path, map_location='cpu'), strict=False)
            model_final_fold = configure_model_for_tuning(model_final_fold, args.tuning_strategy, num_classes)

            logger.info(f"--- Addestramento e valutazione del modello finale per il Fold {fold + 1} ---")
            train_loader_final = DataLoader(train_dataset, batch_size=best_params_fold['batch_size'], shuffle=True, drop_last=False, collate_fn=collate_fn)
            val_loader_final = DataLoader(val_dataset, batch_size=best_params_fold['batch_size'], shuffle=False, drop_last=False, collate_fn=collate_fn)
            test_loader = DataLoader(test_dataset, batch_size=best_params_fold['batch_size'], shuffle=False, drop_last=False, collate_fn=collate_fn)

            logger.info(f"\n=== DEBUG DATALOADER FOLD {fold + 1} ===")
            logger.info(f"Train dataset size: {len(train_dataset)}")
            logger.info(f"Val dataset size: {len(val_dataset)}")
            logger.info(f"Test dataset size: {len(test_dataset)}")

            logger.info(f"\n=== DEBUG DATALOADER FOLD {fold + 1} ===")
            logger.info(f"Train dataset size: {len(train_dataset)}")
            logger.info(f"Val dataset size: {len(val_dataset)}")
            logger.info(f"Test dataset size: {len(test_dataset)}")
            

            # Conta quanti batch effettivi
            train_batches = len(train_loader_final)
            val_batches = len(val_loader_final)
            test_batches = len(test_loader)

            # CALCOLO CORRETTO delle finestre usate (con drop_last=False usa TUTTE)
            train_samples_used = len(train_dataset)
            val_samples_used = len(val_dataset)
            test_samples_used = len(test_dataset)

            logger.info(f"Train batches: {train_batches} (finestre usate: {train_samples_used}/{len(train_dataset)})")
            logger.info(f"Val batches: {val_batches} (finestre usate: {val_samples_used}/{len(val_dataset)})")
            logger.info(f"Test batches: {test_batches} (finestre usate: {test_samples_used}/{len(test_dataset)})")

            logger.info("Inizio training finale per il fold con i migliori iperparametri...")

            train_with_cm.train(net=model_final_fold, 
                train_loader=train_loader_final, 
                val_loader=val_loader_final,
                exp_figures_dir=exp_figures_dir,
                exp_reports_dir=exp_reports_dir,
                epochs=args.epochs, 
                lr=best_params_fold['lr'],
                figure_name=f"kfold_training_fold_{fold+1}_{exp_name}", 
                save_plots=True
            )

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
                labels_dict=labels_dict
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
            class_names=[labels_dict.get(i) for i in range(num_classes)],
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
        train_with_cm.train(net=final_model, 
            train_loader=final_train_loader, 
            val_loader=None, 
            exp_figures_dir=exp_figures_dir,
            exp_reports_dir=exp_reports_dir,
            epochs=args.epochs, 
            lr=best_hyperparameters['lr'],
            figure_name="final_training_all_data",
            save_plots=False
        )

        
        # Salva il modello finale.
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
            labels_dict=labels_dict
        )


    
if __name__ == "__main__":
    args = parse_args()
    main(args)