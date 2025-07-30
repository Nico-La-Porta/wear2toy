import os
import sys
import argparse
import shutil
import pandas as pd
import numpy as np
import glob
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from collections import Counter
from collections import defaultdict

# Import da Optuna
import optuna
from optuna.pruners import MedianPruner

# Import custom modules
from app_config import REPORTS_DIR, FIGURES_DIR, MODELS_DIR
import sliding_window_on_data
from models.DeepConvLSTM import DeepConvLSTM, HARDataset
import train_with_cm
import normalization
from utils.log_config import logger
from figures import plot_CM, combine_kfold_confusion_matrices
from utils import mapping_activity
from utils.transformations import *
from utils.transformations_utils import *


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
                        help='Strategia di fine-tuning: "lp" (linear probing con testa a singolo layer) o "fft" (full fine-tuning con testa sequenziale).')
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
        logger.info("Costruzione di una testa a singolo layer (nn.Linear).")
        model.classification_head = nn.Linear(n_hidden, num_classes)
        
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
    logger.info(f"===== INIZIO ESPERIMENTO: {exp_name} =====")

    data_path = "C:\\codes\\HumanActivityRecognition\\data\\downstream_data"
    df_toy, toy_mapping = load_and_preprocess_data(args.toy, data_path)
    logger.info(f"Classi presenti dopo il filtraggio: {df_toy['action_id'].unique()}")

    # --- Hold-out set per bambini specifici ---
    df_train_val = df_toy.copy()
    if args.holdout_kids:
        logger.info(f"Separazione dei bambini per il test hold-out: {args.holdout_kids}")
        df_test_holdout = df_toy[df_toy['kid_id'].isin(args.holdout_kids)]
        df_train_val = df_toy[~df_toy['kid_id'].isin(args.holdout_kids)]
        logger.info(f"Dimensioni Training/Validation set: {df_train_val.shape}")
        logger.info(f"Dimensioni Hold-out Test set: {df_test_holdout.shape}")
    
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
    
    unique_labels = np.unique(Y)
    label_mapping = {label: i for i, label in enumerate(unique_labels)}
    Y_mapped = np.array([label_mapping[y] for y in Y])
    
    num_classes = len(unique_labels)
    labels_dict = toy_mapping["encoded_to_name"]
    class_names = [labels_dict.get(i, f"Unknown {i}") for i in unique_labels]

    logger.info(f"Dati di training/validation pronti: X.shape={X.shape}, num_classes={num_classes}")

    skf = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=args.random_state)
    all_fold_scores = []
    
    exp_models_dir = os.path.join(MODELS_DIR, exp_name)
    os.makedirs(exp_models_dir, exist_ok=True)
    
    pretrained_model_path = get_pretrained_model_path(args.pt_norm, args.pt_aug)
    logger.info(f"Utilizzo del modello pre-addestrato: {pretrained_model_path}")

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
        # --- FINE BLOCCO DI DEBUG ---

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
            X_train_fold = np.concatenate([X_train_fold, X_train_aug])
            Y_train_fold = np.concatenate([Y_train_fold, Y_train_fold.copy()])

        train_dataset = HARDataset(X_train_fold, Y_train_fold, train_indices_meta, train_consec)
        val_dataset = HARDataset(X_val_fold, Y_val_fold, val_indices_meta, val_consec)
        test_dataset = HARDataset(X_test_fold, Y_test_fold, test_indices_meta, test_consec)


        def objective_fold(trial):
            lr = trial.suggest_float('lr', 1e-4, 1e-1, log=True)
            batch_size = trial.suggest_categorical('batch_size', [2, 4])
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
            
            model = DeepConvLSTM()
            model.load_state_dict(torch.load(pretrained_model_path, map_location='cpu'), strict=False)
            model = configure_model_for_tuning(model, args.tuning_strategy, num_classes)

            best_f1 = train_with_cm.train(model, train_loader, val_loader, epochs=args.epochs, batch_size=batch_size, lr=lr)
            return best_f1

        study = optuna.create_study(direction='maximize', pruner=MedianPruner())
        study.optimize(objective_fold, n_trials=args.n_trials)
        
        best_params = study.best_params
        best_f1 = study.best_value
        logger.info(f"Migliori iperparametri per il fold {fold + 1}: {best_params} (F1-score: {best_f1:.4f})")

        model_final_fold = DeepConvLSTM()
        model_final_fold.load_state_dict(torch.load(pretrained_model_path, map_location='cpu'), strict=False)
        model_final_fold = configure_model_for_tuning(model_final_fold, args.tuning_strategy, num_classes)
        
        train_loader_final = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True, drop_last=True)
        val_loader_final = DataLoader(val_dataset, batch_size=best_params['batch_size'], shuffle=False, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=best_params['batch_size'], shuffle=False, drop_last=True)

        train_with_cm.train(model_final_fold, train_loader_final, val_loader_final, epochs=args.epochs, **best_params)
        
        _, _, test_f1 = train_with_cm.evaluate_model(
            model_final_fold, test_loader,
            figure_name=f"cm_test_fold_{fold + 1}_{exp_name}",
            save_confusion_matrix=True, save_predictions_csv=True, save_f1_score=True,
            labels_dict=labels_dict
        )

        model_path = os.path.join(exp_models_dir, f"best_model_fold_{fold + 1}.pkl")
        torch.save(model_final_fold.state_dict(), model_path)

        all_fold_scores.append({'fold': fold + 1, 'params': best_params, 'val_f1': best_f1, 'test_f1': test_f1, 'model_path': model_path})

    logger.info("\n===== RISULTATI K-FOLD CROSS-VALIDATION =====")
    test_f1_scores = [res['test_f1'] for res in all_fold_scores]
    mean_f1 = np.mean(test_f1_scores)
    std_f1 = np.std(test_f1_scores)
    
    logger.info(f"F1-score medio sui test fold: {mean_f1:.4f} ± {std_f1:.4f}")
    
    best_fold = max(all_fold_scores, key=lambda x: x['val_f1'])
    logger.info(f"Miglior fold (basato su val F1): Fold {best_fold['fold']}")
    logger.info(f"  -> Iperparametri: {best_fold['params']}")
    logger.info(f"  -> Val F1: {best_fold['val_f1']:.4f}, Test F1: {best_fold['test_f1']:.4f}")

    results_df = pd.DataFrame(all_fold_scores)
    results_df.to_csv(os.path.join(REPORTS_DIR, f'kfold_results_{exp_name}.csv'), index=False)

if __name__ == "__main__":
    args = parse_args()
    main(args)