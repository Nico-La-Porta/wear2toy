import os
import pandas as pd
import numpy as np
from app_config import REPORTS_DIR, FIGURES_DIR, MODELS_DIR
import sliding_window_on_data
from torch.utils.data import DataLoader
import glob
from collections import defaultdict

from models.DeepConvLSTM import DeepConvLSTM, HARDataset, collate_fn
import optuna
import optuna.visualization as vis
import train
import torch
import train_with_cm

import matplotlib.pyplot as plt
#definisco il path da cui leggere i .csv

from utils.log_config import logger
from utils.data_preprocessing import  get_class_distribution

from optuna.pruners import MedianPruner

from figures import plot_CM, combine_kfold_confusion_matrices
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from utils import mapping_activity

#definisco il path da cui leggere i .csv

path="C:\codes\HumanActivityRecognition\data\downstream_data"
logger.debug(path)

#CREO SOTTOCARTELLA NELLA CARTELLA FIGURES_DIR CHE SI CHIAMA spoon_figures_lp
figures_spoon_path = os.path.join(FIGURES_DIR, 'spoon_figures_LP')
os.makedirs(figures_spoon_path, exist_ok=True)  # Crea la cartella




#Visto che l'informazione relativa all'id del bambino lo abbiamo, così come abbiamo anche l'informazione relativa
#al giocattolo, vado a filtrare i .csv in modo tale da avere solo le righe che hanno l'attività non nulla e poi
#appendo tutte le righe delle righe non nulle in un unico dataframe
#per tutti i file che terminano in .csv nella cartella path
# Definisco il percorso della cartella contenente i CSV

# Nome del file CSV finale
final_csv_path = os.path.join(path, 'df_SPOON_non_null.csv')

# Controllo se il file esiste già
if os.path.exists(final_csv_path):
    logger.debug(f"Il file {final_csv_path} esiste già. Lo sto caricando...")
    df_spoon = pd.read_csv(final_csv_path)
else:
    #trovo tutti i file che corrispondono a "DO" nella cartella path e li stampo a schermo
    files = glob.glob(os.path.join(path, "*_SP*.csv"))
    logger.debug(f"Files: {files}")
    
    kid_spoon, kid_spoon_no_null = [], [] # liste per salvare utenti prima e dopo il merge 
    df_list_spoon = [] # lista vuota per appendere i dataframe con attività non nulla

    for file in files:
        df = pd.read_csv(file)
        logger.debug(f"Original shape: {df.shape}")
        kid_spoon.append(df['kid_id'].unique())
        df = df[df['action_id'] != 0] #filtro le righe con action_id non nullo
        logger.debug(f"Filtered shape: {df.shape}")
        kid_spoon_no_null.append(df['kid_id'].unique())

        logger.debug(f"Columns: {df.columns}")
        logger.debug(f"Action counts:\n{df['action'].value_counts()}")
        logger.debug(f"Toy counts:\n{df['toy_id'].value_counts()}")
        logger.debug("="*50)  # Separatore tra i file
    
    # mi stampo gli utenti prima di fare il merge e dopo il merge
    if len(kid_spoon) > 0:
        logger.info(f"Numero di utenti che hanno fatto almeno un azione: {len(kid_spoon_no_null)/len(kid_spoon)}")
    else:
        logger.info("Nessun utente trovato nei file analizzati (kid_spoon è vuoto).")

    '''
    Visto che l'informazione relativa all'id del bambino lo abbiamo, così come abbiamo anche l'informazione relativa
    al giocattolo, vado a filtrare i .csv in modo tale da avere solo le righe che hanno l'attività non nulla e poi
    appendo tutte le righe non nulle in un unico dataframe per tutti i file che terminano in .csv nella cartella path
    '''

    for file in os.listdir(path):
        if not file.endswith('.csv'):
            continue
    
        #leggo solo i file che dopo l'undescore ha SP*.csv
        if file.split('_')[-1].startswith('SP') and file.endswith('.csv'): #controllo che il file termini con .csv
            df_temp=pd.read_csv(os.path.join(path,file))
            df_temp = df_temp[df_temp['action_id'] != 0]
            df_list_spoon.append(df_temp)

    df_spoon = pd.concat(df_list_spoon)
    logger.info(f"Shape finale del dataframe: {df_spoon.shape}")
    logger.info(f"Colonne del dataframe: {list(df_spoon.columns)}")
    logger.info(f"Conteggio delle azioni:\n{df_spoon['action'].value_counts()}")

    # salvo il dataframe
    df_spoon.to_csv(os.path.join(path,'df_SPOON_non_null.csv'),index=False)


#ora divido il dataframe in base all'attività (action_id) e salvo i dataframe in un file .csv
#per ogni attività

#filtro le classi per trovare solo quelle che mi interessano
logger.info("FILTRAGGIO CLASSI DAL DATASET SPOON")

# Mostra distribuzione originale
logger.info("Distribuzione action_id originale:")
original_counts = df_spoon['action_id'].value_counts().sort_index()
for action_id, count in original_counts.items():
    logger.info(f"   Action {action_id}: {count} righe")


# Rimuovo le classi che non mi interessano
classes_to_remove = [4]
logger.info(f"Rimozione action_id: {classes_to_remove}")

df_spoon_filtered = df_spoon[~df_spoon['action_id'].isin(classes_to_remove)] # filtro il dataframe per rimuovere le classi che non mi interessano

logger.info(f"Righe prima del filtro: {len(df_spoon)}")
logger.info(f"Righe dopo il filtro: {len(df_spoon_filtered)}")
logger.info(f"Righe rimosse: {len(df_spoon) - len(df_spoon_filtered)}")


logger.info("Distribuzione action_id dopo filtro:")
filtered_counts = df_spoon_filtered['action_id'].value_counts().sort_index()
for action_id, count in filtered_counts.items():
    logger.info(f"   Action {action_id}: {count} righe")


#ora divido il dataframe in base all'attività (action_id) e salvo i dataframe in un file .csv
#per ogni attività

for action_id in df_spoon_filtered['action_id'].unique():
    df_action = df_spoon_filtered[df_spoon_filtered["action_id"] == action_id] #filtro il dataframe in base all'attività
    logger.debug(f"Dimensioni del dataframe df_spoon_action_{action_id} - {df_action.shape}") #log delle dimensioni del dataframe
    logger.info(f"Conteggio delle attività per df_spoon_action_{action_id}") #log del conteggio delle attività
    #salvo il dataframe
    df_action.to_csv(os.path.join(path,f'df_spoon_action_{action_id}.csv'),index=False) #index=False per non salvare l'indice
    logger.debug(f"Salvato il dataframe df_spoon_action_{action_id}.csv")


#applico sliding window con la funzion process_csv
nb_sensor_channels = 9
sliding_window_length = 100
sliding_window_step = 50

#ora applico la funzione sliding window (che mi da come output x_window e y_window) a tutti i .csv relativi al giocattolo spoon
#e poi concateno tutto in un unica x e y 

X, Y = [], []
kid_action_counts = {}
all_consecutivity = []
all_window_indices = []
global_window_id = 0

for action_file in [f for f in os.listdir(path) if f.endswith('.csv') and f.startswith('df_spoon_action_') and not f.endswith('normalized_pt.csv')]:
    file_path = os.path.join(path, action_file)
    action = action_file.split('_')[-1].split('.')[0]

    X_windows, Y_windows, kid_id_action_dict, is_consecutive = sliding_window_on_data.process_csv(file_path, nb_sensor_channels, sliding_window_length, sliding_window_step)

    
    for i in range(len(X_windows)):
        all_window_indices.append({
            'global_window_id': global_window_id,
            'action_id': action,
            #'local_window_id': i,
            #'file': action_file
        })
        global_window_id += 1
    
    X.append(X_windows)
    Y.append(Y_windows)
    all_consecutivity.extend(is_consecutive)

    logger.info(f"Numero totale di finestre per l'azione {action}: {len(X_windows)}")
    logger.info(f"Finestre consecutive per l'azione {action}: {sum(is_consecutive)}")
    logger.info(f"Finestre non consecutive per l'azione {action}: {len(is_consecutive) - sum(is_consecutive)}")
    logger.info(f"Numero  totale di finestre per l'azione {action}:{len(X_windows)}")
    kid_action_counts[f"spoon_action_{action}"] = kid_id_action_dict #aggiungo il dizionario al dizionario principale per tenere traccia del numero di finestre per ogni bambino per ogni azione, ogni spoon_action è una chiave e il valore è un dizionario con il numero di finestre per ogni bambino
    logger.info(f"Contenuto finale di kid_action_counts: {kid_action_counts}") #per veere quante finestre per ogni azione e per ogni bambino sono state elaborte 


# Concateno tutti i dati in un unico array per X e Y
X = np.concatenate(X, axis=0)
Y = np.concatenate(Y, axis=0)
all_consecutivity = np.array(all_consecutivity)






logger.info(f"Dimensioni di X dopo aggiunta indici finestre: {X.shape}")

logger.info(f"Dimensioni di X dopo concatenazione: {X.shape}")

consecutivity_expanded = all_consecutivity.reshape(-1, 1, 1)  # (n_windows, 1, 1)
consecutivity_repeated = np.repeat(consecutivity_expanded, X.shape[1], axis=1)  # (n_windows, window_length, 1)
# Ora puoi aggiungere le informazioni di consecutività come feature aggiuntiva
X_with_consecutivity = np.concatenate([X, consecutivity_repeated], axis=2)  # (n_windows, window_length, n_features+1)


# Stampo le dimensioni di X e Y
logger.info(f"Dimensioni di X finale: {X.shape}")
logger.info(f"Dimensioni di Y finale: {Y.shape}")
logger.info(f"Tipo di Y: {Y.dtype}")
logger.info(f"Shape di Y: {Y.shape}")
logger.info(f"Primi 5 elementi di Y: {Y[:5]}")
logger.info(f"Tipo del primo elemento: {type(Y[0])}")
# Log delle statistiche
logger.info(f"Finestre totali: {len(all_consecutivity)}")
logger.info(f"Finestre consecutive: {sum(all_consecutivity)} ({sum(all_consecutivity)/len(all_consecutivity)*100:.1f}%)")
logger.info(f"Finestre non consecutive: {len(all_consecutivity) - sum(all_consecutivity)} ({(len(all_consecutivity) - sum(all_consecutivity))/len(all_consecutivity)*100:.1f}%)")

if Y.ndim > 1:
    logger.info(f"Y ha {Y.ndim} dimensioni ({Y.shape}), lo appiattisco...")
    Y = Y.flatten()  # Converte da (2823, 1) a (2823,)
    logger.info(f"Y dopo flatten: {Y.shape}")

logger.info(f"Shape finale di Y: {Y.shape}")
logger.info(f"Primi 5 elementi di Y: {Y[:5]}")
logger.info(f"Tipo del primo elemento: {type(Y[0])}")

#remap delle etichette per renderle consecutive
unique_labels = np.unique(Y)
label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}

logger.info(f"Mappatura etichette: {label_mapping}")

# Ora il remapping funzionerà correttamente
Y_remapped = np.array([label_mapping[label] for label in Y])

logger.info("Distribuzione finale delle classi con etichette rimappate:")
get_class_distribution(Y_remapped, "Test Dataset con Etichette Rimappate")

#uso le Y_remapped per il resto del codice
Y = Y_remapped


# Usa direttamente il mapping hardcodato
spoon_mapping = mapping_activity.SPOON_ACTION_MAPPING
print(f"Spoon mapping keys: {list(spoon_mapping.keys())}")
labels_dict = spoon_mapping["encoded_to_name"]
print(f"Labels dict: {labels_dict}")


class_names = [spoon_mapping["encoded_to_name"][i] for i in range(3)]
print(f"Class names: {class_names}")


logger.info(f"\n=== DEBUG DATASET ===")
logger.info(f"Numero totale finestre: {len(X)}")
logger.info(f"Shape di X: {X.shape}")
logger.info(f"Shape di Y: {Y.shape}")
logger.info(f"Distribuzione classi: {np.bincount(Y)}")

# Commenta fino alla fine del codice
# Parametri per K-Fold
K_FOLDS = 3
random_state = 42

# PRIMA del K-Fold (riga ~270), aggiungi:
logger.info(f"\n=== DEBUG INDICI PRIMA DEL K-FOLD ===")
logger.info(f"Lunghezza all_window_indices: {len(all_window_indices)}")

# Controlla duplicati nei global_window_id
global_ids = [w['global_window_id'] for w in all_window_indices]
unique_global_ids = set(global_ids)
logger.info(f"Global IDs totali: {len(global_ids)}")
logger.info(f"Global IDs unici: {len(unique_global_ids)}")

if len(global_ids) != len(unique_global_ids):
    logger.error("DUPLICATI trovati in all_window_indices!")
    duplicates = [id for id in global_ids if global_ids.count(id) > 1]
    logger.error(f"IDs duplicati: {set(duplicates)}")
    
    # Trova dove sono i duplicati
    for dup_id in set(duplicates):
        positions = [i for i, w in enumerate(all_window_indices) if w['global_window_id'] == dup_id]
        logger.error(f"Global ID {dup_id} trovato alle posizioni: {positions}")
        logger.error(f"  Posizione {positions[0]}: {all_window_indices[positions[0]]}")
        logger.error(f"  Posizione {positions[1]}: {all_window_indices[positions[1]]}")
else:
    logger.info("Nessun duplicato in all_window_indices")

logger.info(f"Primi 10 global_ids: {global_ids[:10]}")
logger.info(f"Ultimi 10 global_ids: {global_ids[-10:]}")

# Creo il StratifiedKFold
skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=random_state)

#CONTROLLA I FOLD E GLI INDICI (14 train e 4 per val), 

# Dizionario per salvare i risultati di ogni fold
fold_results = {
    'fold': [],
    'best_f1_score': [],
    'best_params': [],
    'test_f1_score': []
}

# Percorsi per i modelli
BEST_MODEL_KFOLD_DIR = os.path.join(MODELS_DIR, "kfold_spoon_models_3_folds_LP")
os.makedirs(BEST_MODEL_KFOLD_DIR, exist_ok=True)


logger.info("=== INIZIO STRATIFIED K-FOLD CROSS-VALIDATION ===")
logger.info(f"Numero di fold: {K_FOLDS}")


# Lista per salvare tutti i punteggi per la selezione finale
all_fold_scores = []


for fold, (train_val_idx, test_idx) in enumerate(skf.split(X, Y)):
    logger.info(f"\n=== FOLD {fold + 1}/{K_FOLDS} ===")
    logger.info(f"\n=== FOLD {fold + 1} SPLIT DEBUG ===")
    logger.info(f"Train+Val indices: {len(train_val_idx)} finestre")
    logger.info(f"Test indices: {len(test_idx)} finestre")
    logger.info(f"Test indices: {test_idx}")

    logger.info(f"Train+Val indices (first 10): {train_val_idx[:10]}")
    logger.info(f"Test indices: {test_idx}")
    
    # Verifica che non ci siano sovrapposizioni
    overlap = set(train_val_idx).intersection(set(test_idx))
    if overlap:
        logger.error(f"SOVRAPPOSIZIONE trovata: {overlap}")
    else:
        logger.info("Nessuna sovrapposizione train-test")
    
    # Verifica gli indici delle finestre corrispondenti
    test_window_ids = [all_window_indices[i] for i in test_idx]
    train_val_window_ids = [all_window_indices[i] for i in train_val_idx]
    
    logger.info(f"Test window global_ids: {[w['global_window_id'] for w in test_window_ids]}")
    logger.info(f"Train+Val window global_ids (first 10): {[w['global_window_id'] for w in train_val_window_ids[:10]]}")
    
    # Split dei dati per questo fold
    X_train_val, X_test_fold = X[train_val_idx], X[test_idx]
    Y_train_val, Y_test_fold = Y[train_val_idx], Y[test_idx]

    
    
   # GESTIONE CORRETTA DEGLI INDICI
    train_val_indices = [all_window_indices[i] for i in train_val_idx] # estraggo per ogni indice nel set di train_val gli indici della finestra originale 
    test_fold_indices = [all_window_indices[i] for i in test_idx] # estraggo per ogni indice nel set di test gli indici della finestra originale
    train_val_consecutivity = all_consecutivity[train_val_idx] # estraggo per ogni indice nel set di train_val le informazioni di consecutività
    test_fold_consecutivity = all_consecutivity[test_idx] # estraggo per ogni indice nel set di test le informazioni di consecutività

    logger.info(f"Train+Val indices: {len(train_val_indices)} finestre")
    logger.info(f"Test indices: {len(test_fold_indices)} finestre")
    
    # Ulteriore split di train_val in train e validation (80-20)
    train_val_split = int(0.8 * len(X_train_val))
    train_fold_size = train_val_split
    val_fold_size = len(train_val_idx) - train_val_split
    
    logger.info(f"  → Train: {train_fold_size} finestre")
    logger.info(f"  → Val: {val_fold_size} finestre")
    
    X_train_fold = X_train_val[:train_val_split]
    Y_train_fold = Y_train_val[:train_val_split]
    X_val_fold = X_train_val[train_val_split:]
    Y_val_fold = Y_train_val[train_val_split:]
    
    # SPLIT DEGLI INDICI E CONSECUTIVITÀ
    train_indices_fold = train_val_indices[:train_val_split]
    val_indices_fold = train_val_indices[train_val_split:]
    train_consecutivity_fold = train_val_consecutivity[:train_val_split]
    val_consecutivity_fold = train_val_consecutivity[train_val_split:]

    logger.info(f"Train indices: {len(train_indices_fold)} finestre")
    logger.info(f"Val indices: {len(val_indices_fold)} finestre")
    logger.info(f"Test indices: {len(test_fold_indices)} finestre")
    
    
    # Creo i dataset per questo fold CON INDICI E CONSECUTIVITÀ
    train_dataset_fold = HARDataset(X_train_fold, Y_train_fold, train_indices_fold, train_consecutivity_fold)
    val_dataset_fold = HARDataset(X_val_fold, Y_val_fold, val_indices_fold, val_consecutivity_fold)
    test_dataset_fold = HARDataset(X_test_fold, Y_test_fold, test_fold_indices, test_fold_consecutivity)


    # Ottimizzazione degli iperparametri per questo fold
    def objective_fold(trial):
        lr = trial.suggest_float('lr', 1e-4, 1e-1, log=True)
        batch_size = trial.suggest_categorical('batch_size', [2, 4])
        
        # Creo i DataLoader con collate_fn personalizzato
        train_loader = DataLoader(train_dataset_fold, batch_size=batch_size, shuffle=True, 
                                drop_last=False, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset_fold, batch_size=batch_size, shuffle=False, 
                              drop_last=False, collate_fn=collate_fn)
        
        # Creo il modello
        model = DeepConvLSTM()
        
        # Carico i pesi pre-addestrati
        model.load_state_dict(
            torch.load(r'C:\codes\HumanActivityRecognition\models\best_model_dl_without_anything.pkl', 
                      map_location=torch.device('cpu')), 
            strict=False
        )
        
        # Sostituisco la testa
        num_ftrs = model.classification_head.in_features
        model.classification_head = nn.Linear(num_ftrs, 3)
        model.set_n_classes(3)

        """for param in model.parameters():
            param.requires_grad = True"""
        
        # FREEZO TUTTI I PARAMETRI ECCETTO LA TESTA (LINEAR PROBING)
        for name, param in model.named_parameters():
            if 'classification_head' not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
        
        # Training SENZA salvare i risultati per i trial intermedi
        best_f1_score = train_with_cm.train(model, train_loader, val_loader, 
                                           epochs=100, batch_size=batch_size, lr=lr,
                                           figure_name=f"fold_{fold+1}_trial_{trial.number}",
                                           save_final_results=False, is_best_trial=False)
        
        return best_f1_score
    
    # Ottimizzazione per questo fold
    logger.info(f"Inizio ottimizzazione iperparametri per fold {fold + 1}")
    study_fold = optuna.create_study(
        direction='maximize',
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )
    study_fold.optimize(objective_fold, n_trials=50)

    # Salvo i migliori iperparametri per questo fold
    best_params_fold = study_fold.best_params
    best_f1_fold = study_fold.best_value

    logger.info(f"Fold {fold + 1} - Migliori iperparametri: {best_params_fold}")
    logger.info(f"Fold {fold + 1} - Miglior F1 score: {best_f1_fold}")

    # Addestro il modello finale per questo fold con i migliori iperparametri
    best_lr = best_params_fold['lr']
    best_batch_size = best_params_fold['batch_size']

    # Ricreo il modello con i migliori iperparametri
    model_final_fold = DeepConvLSTM()
    model_final_fold.load_state_dict(
        torch.load(r'C:\codes\HumanActivityRecognition\models\best_model_dl_without_anything.pkl', 
                  map_location=torch.device('cpu')), 
        strict=False
    )

    # Sostituisco la testa
    num_ftrs = model_final_fold.classification_head.in_features
    model_final_fold.classification_head = nn.Linear(num_ftrs, 3)
    model_final_fold.set_n_classes(3)

    """for param in model_final_fold.parameters():
        param.requires_grad = True"""
    
    # FREEZO TUTTI I PARAMETRI ECCETTO LA TESTA (LINEAR PROBING)
    for name, param in model_final_fold.named_parameters():
        if 'classification_head' not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

    # Training finale per questo fold
    # Training finale per questo fold con i migliori iperparametri
    train_loader_final = DataLoader(train_dataset_fold, batch_size=best_batch_size, shuffle=True, 
                                  drop_last=False, collate_fn=collate_fn)
    val_loader_final = DataLoader(val_dataset_fold, batch_size=best_batch_size, shuffle=False, 
                                drop_last=False, collate_fn=collate_fn)
    
    test_loader_fold = DataLoader(test_dataset_fold, batch_size=best_batch_size, shuffle=False, 
                                drop_last=False, collate_fn=collate_fn)
    
    
    
    
    
    logger.info(f"\n=== DEBUG DATALOADER FOLD {fold + 1} ===")
    logger.info(f"Train dataset size: {len(train_dataset_fold)}")
    logger.info(f"Val dataset size: {len(val_dataset_fold)}")
    logger.info(f"Test dataset size: {len(test_dataset_fold)}")
    logger.info(f"Batch size: {best_batch_size}")

    logger.info(f"\n=== DEBUG DATALOADER FOLD {fold + 1} ===")
    logger.info(f"Train dataset size: {len(train_dataset_fold)}")
    logger.info(f"Val dataset size: {len(val_dataset_fold)}")
    logger.info(f"Test dataset size: {len(test_dataset_fold)}")
    logger.info(f"Batch size: {best_batch_size}")

    # Conta quanti batch effettivi
    train_batches = len(train_loader_final)
    val_batches = len(val_loader_final)
    test_batches = len(test_loader_fold)

    # CALCOLO CORRETTO delle finestre usate (con drop_last=False usa TUTTE)
    train_samples_used = len(train_dataset_fold)
    val_samples_used = len(val_dataset_fold)
    test_samples_used = len(test_dataset_fold)

    logger.info(f"Train batches: {train_batches} (finestre usate: {train_samples_used}/{len(train_dataset_fold)})")
    logger.info(f"Val batches: {val_batches} (finestre usate: {val_samples_used}/{len(val_dataset_fold)})")
    logger.info(f"Test batches: {test_batches} (finestre usate: {test_samples_used}/{len(test_dataset_fold)})")
    
    # Training finale CON salvataggio dei risultati
    final_f1_score = train_with_cm.train(
        model_final_fold, train_loader_final, val_loader_final,
        epochs=100, batch_size=best_batch_size, lr=best_lr,
        figure_name=f"kfold_training_fold_{fold+1}",  # NON CONTIENE FINAL
        save_final_results=False,  #NON SALVO
        is_best_trial=True,
        figures_dir=figures_spoon_path, 
        reports_dir=figures_spoon_path
    )
    

    logger.info(f"\n=== PRE-EVALUATE DEBUG ===")
    logger.info(f"Test dataset size prima di evaluate: {len(test_dataset_fold)}")
    logger.info(f"Test loader size prima di evaluate: {len(test_loader_fold)}")

    # Testa il primo batch
    for i, batch in enumerate(test_loader_fold):
        if i == 0:
            inputs, targets, indices, consecutivity = batch
            logger.info(f"Primo batch - inputs shape: {inputs.shape}")
            logger.info(f"Primo batch - targets shape: {targets.shape}")
            logger.info(f"Primo batch - indices: {indices}")
            break
    test_loss, test_acc, test_f1_fold = train_with_cm.evaluate_model(
        model_final_fold, test_loader_fold, 
        figure_name=f"cm_test_fold_{fold + 1}_kfold_inference_3_folds_Tspoon_PTNnone_PTAnone_FTNnone_FTAnone",
        save_confusion_matrix=True, 
        save_predictions_csv=True,
        save_f1_score=True,
        labels_dict=labels_dict,
        figures_dir=figures_spoon_path,
        reports_dir=figures_spoon_path
    )

    # Salvo il modello di questo fold
    model_path_fold = os.path.join(BEST_MODEL_KFOLD_DIR, f"best_model_fold_{fold + 1}_Tspoon_PTNnone_PTAnone_FTNnone_FTAnone.pkl")
    torch.save(model_final_fold.state_dict(), model_path_fold)


    # Salvo i risultati
    fold_results['fold'].append(fold + 1)
    fold_results['best_f1_score'].append(best_f1_fold)
    fold_results['best_params'].append(best_params_fold)
    fold_results['test_f1_score'].append(test_f1_fold)
    
    # Aggiungo alla lista per la selezione finale
    all_fold_scores.append({
        'fold': fold + 1,
        'params': best_params_fold,
        'val_f1': best_f1_fold,
        'test_f1': test_f1_fold,
        'model_path': model_path_fold
    })
    
    logger.info(f"Fold {fold + 1} completato - Test F1: {test_f1_fold:.4f}")



logger.info("\n=== SELEZIONE DEL MIGLIOR MODELLO ===")

# Calcolo statistiche sui fold
fold_f1_scores = [result['test_f1'] for result in all_fold_scores]
mean_f1 = np.mean(fold_f1_scores)
std_f1 = np.std(fold_f1_scores)

logger.info(f"F1 score medio sui {K_FOLDS} fold: {mean_f1:.4f} ± {std_f1:.4f}")
logger.info(f"F1 score per fold: {fold_f1_scores}")

# Seleziono il modello con il miglior F1 score di validazione medio
best_fold_idx = np.argmax([result['val_f1'] for result in all_fold_scores])
best_fold_result = all_fold_scores[best_fold_idx]

logger.info(f"Miglior fold: {best_fold_result['fold']}")
logger.info(f"Migliori iperparametri: {best_fold_result['params']}")
logger.info(f"F1 score validazione: {best_fold_result['val_f1']:.4f}")
logger.info(f"F1 score test: {best_fold_result['test_f1']:.4f}")

# Salvo i risultati di tutti i fold
results_df = pd.DataFrame(fold_results)
results_df.to_csv(os.path.join(figures_spoon_path, 'kfold_results_inference_spoon_Tspoon_PTNnone_PTAnone_FTNnone_FTAnone.csv'), index=False)

# Salvo i migliori iperparametri
best_hyperparameters = best_fold_result['params']
best_hyperparameters['mean_test_f1'] = mean_f1
best_hyperparameters['std_test_f1'] = std_f1
best_hyperparameters['best_fold'] = best_fold_result['fold']

best_hyperparameters_df = pd.DataFrame([best_hyperparameters])
best_hyperparameters_df.to_csv(
    os.path.join(figures_spoon_path, 'best_hyperparameters_kfold_inference_spoon_Tspoon_PTNnone_PTAnone_FTNnone_FTAnone.csv'), 
    index=False
)

# === PARTE 5: TRAINING FINALE SU TUTTI I DATI ===

logger.info("\n=== TRAINING FINALE SU TUTTI I DATI ===")

# Uso i migliori iperparametri per il training finale
best_lr_final = best_hyperparameters['lr']
best_batch_size_final = best_hyperparameters['batch_size']

# Creo dataset con tutti i dati
final_dataset = HARDataset(
    X, Y, 
    window_indices=all_window_indices,
    consecutivity=all_consecutivity,
    class_names=class_names
)

# Split finale 85-15 per train-val
train_size = int(0.8 * len(final_dataset))
val_size = len(final_dataset) - train_size
final_train_dataset, final_val_dataset = torch.utils.data.random_split(
    final_dataset, [train_size, val_size]
)

final_train_loader = DataLoader(
    final_train_dataset, 
    batch_size=best_batch_size_final, 
    shuffle=True, 
    collate_fn=collate_fn
)

final_val_loader = DataLoader(
    final_val_dataset, 
    batch_size=best_batch_size_final, 
    shuffle=False, 
    collate_fn=collate_fn
)


# Creo modello finale
model_final = DeepConvLSTM()
model_final.load_state_dict(
    torch.load(r'C:\codes\HumanActivityRecognition\models\best_model_dl_without_anything.pkl', 
              map_location=torch.device('cpu')), 
    strict=False
)

# Sostituisco la testa
num_ftrs = model_final.classification_head.in_features
model_final.classification_head = nn.Linear(num_ftrs, 3)
model_final.set_n_classes(3)

# FREEZO TUTTI I PARAMETRI ECCETTO LA TESTA (LINEAR PROBING)
for name, param in model_final.named_parameters():
    if 'classification_head' not in name:
        param.requires_grad = False
    else:
        param.requires_grad = True

# Training finale
# Training finale
# Training finale CON salvataggio dei risultati finali
logger.info("Inizio training finale su tutti i dati...")
final_f1_score = train_with_cm.train(
    model_final, final_train_loader, final_val_loader,
    epochs=100, 
    batch_size=best_batch_size_final, 
    lr=best_lr_final,
    figure_name="final_training_all_data",  # Nome che contiene "final" ma NON "fold"
    save_final_results=True,  # QUI SALVO 
    is_best_trial=True,
    figures_dir=figures_spoon_path, 
    reports_dir=figures_spoon_path
)

# Salvo il modello finale
FINAL_MODEL_PATH = os.path.join(BEST_MODEL_KFOLD_DIR , "best_model_kfold_final_inference_spoon_Tspoon_PTNnone_PTAnone_FTNnone_FTAnone.pkl")
torch.save(model_final.state_dict(), FINAL_MODEL_PATH)

logger.info(f"Modello finale salvato: {FINAL_MODEL_PATH}")
logger.info(f"F1 score finale: {final_f1_score:.4f}")

# === PARTE 6: VALUTAZIONE FINALE ===

# Confusion matrix su tutti i dati
plot_CM(
    mdl_class=DeepConvLSTM,
    mdl_weights=FINAL_MODEL_PATH,
    X=X,
    Y=Y,
    batch_size=best_batch_size_final,
    figure_name="cm_final_kfold_inference_spoon_all_data_Tspoon_PTNnone_PTAnone_FTNnone_FTAnone",
    labels_dict=labels_dict,
    figures_dir=figures_spoon_path,
    reports_dir=figures_spoon_path,
)

logger.info("=== PROCEDURA K-FOLD COMPLETATA ===")
logger.info(f"Migliori iperparametri: {best_hyperparameters}")
logger.info(f"F1 score medio K-Fold: {mean_f1:.4f} ± {std_f1:.4f}")
logger.info(f"F1 score finale: {final_f1_score:.4f}")





#AGGREGAZIONE DELLE CM DI TEST

cm_combined, y_true, y_pred = combine_kfold_confusion_matrices(
    fold_results_dir=  figures_spoon_path,
    num_folds=3,
    toy_name="spoon",
    save_path="combined_cm_Tspoon_PTNnone_PTAnone_FTNnone_FTAnone.png",
    class_names=class_names,
    figures_dir= figures_spoon_path,
)

