import os
import pandas as pd
import numpy as np
from app_config import REPORTS_DIR, FIGURES_DIR, MODELS_DIR
import sliding_window_on_data
from torch.utils.data import DataLoader
import glob
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
import seaborn as sns

from models.DeepConvLSTM import DeepConvLSTM, HARDataset 
import optuna
from optuna.pruners import MedianPruner
import optuna.visualization as vis
import train
import torch
import torch.nn as nn
import train_with_cm
import matplotlib.pyplot as plt
#definisco il path da cui leggere i .csv

from utils.log_config import logger
from figures import plot_CM
import normalization

#definisco il path da cui leggere i .csv

path="C:\codes\HumanActivityRecognition\data\downstream_data"
logger.debug(path)


#Visto che l'informazione relativa all'id del bambino lo abbiamo, così come abbiamo anche l'informazione relativa
#al giocattolo, vado a filtrare i .csv in modo tale da avere solo le righe che hanno l'attività non nulla e poi
#appendo tutte le righe delle righe non nulle in un unico dataframe
#per tutti i file che terminano in .csv nella cartella path
# Definisco il percorso della cartella contenente i CSV

# Nome del file CSV finale
final_csv_path = os.path.join(path, 'df_CAR_non_null.csv')

# Controllo se il file esiste già
if os.path.exists(final_csv_path):
    logger.debug(f"Il file {final_csv_path} esiste già. Lo sto caricando...")
    df_car = pd.read_csv(final_csv_path)
else:
    #trovo tutti i file che corrispondono a "_C" nella cartella path e li stampo a schermo
    files = glob.glob(os.path.join(path, "*_C*.csv"))
    logger.debug(f"File trovati: {files}")
    
    kid_car, kid_car_no_null = [], [] # liste per salvare utenti prima e dopo il merge 
    df_list_car = [] # lista vuota per appendere i dataframe con attività non nulla

    for file in files:
        df = pd.read_csv(file)
        
        logger.debug(f"Shape originale del file {file}: {df.shape}")
        kid_car.append(df['kid_id'].unique())
        df = df[df['action_id'] != 0] #filtro le righe con action_id non nullo
        logger.debug(f"Shape dopo il filtro del file {file}: {df.shape}")
        kid_car_no_null.append(df['kid_id'].unique())

        logger.debug(f"Colonne del file {file}: {df.columns}")
        logger.debug(f"Conteggio delle azioni nel file {file}:\n{df['action'].value_counts()}")
        logger.debug(f"Conteggio dei giocattoli nel file {file}:\n{df['toy_id'].value_counts()}")
        logger.debug("="*50)

    # mi stampo gli utenti prima di fare il merge e dopo il merge
    logger.info(f"Numero di utenti che hanno fatto almeno un'azione: {len(kid_car_no_null)/len(kid_car):.2f}")

    '''
    Visto che l'informazione relativa all'id del bambino lo abbiamo, così come abbiamo anche l'informazione relativa
    al giocattolo, vado a filtrare i .csv in modo tale da avere solo le righe che hanno l'attività non nulla e poi
    appendo tutte le righe non nulle in un unico dataframe per tutti i file che terminano in .csv nella cartella path
    '''

    for file in os.listdir(path):
        if not file.endswith('.csv'):
            continue
    
        #leggo solo i file che dopo l'undescore ha C*.csv
        if file.split('_')[-1].startswith('C') and file.endswith('.csv'): #controllo che il file termini con .csv
            df_temp=pd.read_csv(os.path.join(path,file))
            df_temp = df_temp[df_temp['action_id'] != 0]
            df_list_car.append(df_temp)

    df_car = pd.concat(df_list_car)
    logger.info(f"Dimensioni del df_car con tutte le attività non nulle: {df_car.shape}")
    logger.info(f"Colonne del df_car:\n{df_car.columns}")
    logger.info(f"Conteggio delle azioni nel df_car:\n{df_car['action'].value_counts()}")

    # salvo il dataframe
    df_car.to_csv(os.path.join(path,'df_CAR_non_null.csv'),index=False)
    logger.info(f"File salvato come {final_csv_path}")


#ora divido il dataframe in base all'attività (action_id) e salvo i dataframe in un file .csv
#per ogni attività

for action_id in df_car['action_id'].unique():
    df_action = df_car[df_car["action_id"] == action_id] #filtro il dataframe in base all'attività
    logger.debug(f"Dimensioni del dataframe df_car_action_{action_id} - {df_action.shape}") #log delle dimensioni del dataframe
    logger.info(f"Conteggio delle attività per df_car_action_{action_id}") #log del conteggio delle attività
    #salvo il dataframe
    df_action.to_csv(os.path.join(path,f'df_car_action_{action_id}.csv'),index=False) #index=False per non salvare l'indice
    logger.debug(f"Salvato il dataframe df_car_action_{action_id}.csv")


#normalizzo i dati 

# Carico mean e std salvati durante il pre-training
logger.debug(f"REPORTS_DIR attuale: {REPORTS_DIR}")
mean_df = pd.read_csv(os.path.join(REPORTS_DIR, 'mean_trs.csv'))
std_df = pd.read_csv(os.path.join(REPORTS_DIR, 'std_trs.csv'))

mean = mean_df['mean'].values
std = std_df['std'].values

logger.info("Parametri di normalizzazione caricati dal pre-training")
logger.debug(f"Mean shape: {mean.shape}, Std shape: {std.shape}")

#ELIMINO  PRIMA E ULTIMA COLONNA RIGA DI MEAN PERCHE NON SONO FEATURE DI INTERESSE (sono timestamp e P)
mean = mean[1:-1] 
std = std[1:-1]  

logger.info("Parametri di normalizzazione caricati dal pre-training")
logger.debug(f"Mean shape: {mean.shape}, Std shape: {std.shape}")

# Normalizzo i dati per ogni action_id
for action_id in df_car['action_id'].unique():
    # Leggo il file CSV per questa azione
    df_action = pd.read_csv(os.path.join(path, f'df_car_action_{action_id}.csv'))
    
    logger.info(f"Normalizzando action_id {action_id} - Shape originale: {df_action.shape}")
    
    # Applico la normalizzazione usando la funzione del modulo normalization
    df_action_normalized = normalization.normalize_dataframe_mean_std(df_action, mean, std)
    
    # Salvo il DataFrame normalizzato
    normalized_file_path = os.path.join(path, f'df_car_action_{action_id}_normalized.csv')
    df_action_normalized.to_csv(normalized_file_path, index=False)
    
    logger.info(f"Salvato file normalizzato: {normalized_file_path}")
    logger.debug(f"Shape del DataFrame normalizzato: {df_action_normalized.shape}")

logger.info("Normalizzazione completata per tutte le azioni")


#applico sliding window con la funzion process_csv
nb_sensor_channels = 9
sliding_window_length = 100
sliding_window_step = 50

#ora applico la funzione sliding window (che mi da come output x_window e y_window) a tutti i .csv relativi al giocattolo car
#e poi concateno tutto in un unica x e y 

X, Y = [], []
kid_action_counts = {}
all_kid_ids = []
normalized_files = [f for f in os.listdir(path) if f.endswith('_normalized.csv') and 'car_action' in f]

logger.info(f"File normalizzati trovati: {normalized_files}")

for action_file in normalized_files:
    file_path = os.path.join(path, action_file)
    action_id = action_file.split('_')[-2]

    logger.info(f"Processando file normalizzato: {action_file}")

    X_windows, Y_windows, kid_id_action_dict,  kid_ids_for_windows = sliding_window_on_data.process_csv(file_path, nb_sensor_channels, sliding_window_length, sliding_window_step)

    X.append(X_windows)
    Y.append(Y_windows)
    all_kid_ids.extend(kid_ids_for_windows)  # Aggiungi i kid_ids


    logger.info(f"Numero  totale di finestre per l'azione {action_id}:{len(X_windows)}")
    kid_action_counts[f"car_action_{action_id}"] = kid_id_action_dict #aggiungo il dizionario al dizionario principale per tenere traccia del numero di finestre per ogni bambino per ogni azione, ogni car_action è una chiave e il valore è un dizionario con il numero di finestre per ogni bambino
    logger.info(f"Contenuto finale di kid_action_counts: {kid_action_counts}") #per veere quante finestre per ogni azione e per ogni bambino sono state elaborte 


# Concateno tutti i dati in un unico array per X e Y
X = np.concatenate(X, axis=0)
Y = np.concatenate(Y, axis=0)
all_kid_ids = np.array(all_kid_ids)

# Stampo le dimensioni di X e Y
logger.info(f"Dimensioni di X finale: {X.shape}")
logger.info(f"Dimensioni di Y finale: {Y.shape}")
logger.info(f"Kid_ids unici: {np.unique(all_kid_ids)}")


logger.info("Pipeline di normalizzazione e sliding window completata!")


# KFOLD CROSS VALIDATION
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut 
from sklearn.metrics import f1_score, accuracy_score
import json

# Appiattisco (300,1) divneta (300,) per Y
Y_flattened = Y.flatten()  
unique_labels = np.unique(Y_flattened) 
label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
logger.info(f"Mapping delle etichette: {label_mapping}")

# Apply mapping to Y
Y_mapped = np.array([label_mapping[y] for y in Y_flattened])  # Usa Y_flattened

logger.info(f"Shape di Y originale: {Y.shape}")
logger.info(f"Shape di Y_flattened: {Y_flattened.shape}")
logger.info(f"Shape di Y_mapped: {Y_mapped.shape}")

# imposto cross validation con 5 fold oppure Leave one OUT
n_splits = 5
use_leave_one_out = False  # SE true significa che uso leave one out e quindi lascio fuori un soggetto intero

if use_leave_one_out:
    cv = LeaveOneGroupOut()
    logger.info("Using Leave-One-Group-Out cross-validation")
else:
    cv = GroupKFold(n_splits=n_splits)
    logger.info(f"Using GroupKFold with {n_splits} splits")

# per memorizzare risultati di ogni fold
fold_results = []
all_fold_hyperparams = []

for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, Y_mapped, groups=all_kid_ids)): #divido il dataset in train e test usando i bambini come gruppi
    logger.info(f"=== FOLD {fold_idx + 1} ===")
    
    # Splitto dati in train e test
    X_train_fold = X[train_idx]
    Y_train_fold = Y_mapped[train_idx]
    X_test_fold = X[test_idx]
    Y_test_fold = Y_mapped[test_idx]
    
    logger.info(f"Fold {fold_idx + 1} - Train: {X_train_fold.shape}, Test: {X_test_fold.shape}")
    logger.info(f"Kids in train: {np.unique(all_kid_ids[train_idx])}")
    logger.info(f"Kids in test: {np.unique(all_kid_ids[test_idx])}")
    
    # Fsuddivido il train in train e validation (80% train, 20% validation)
    val_split = int(0.8 * len(X_train_fold))
    X_train_cv = X_train_fold[:val_split]
    Y_train_cv = Y_train_fold[:val_split]
    X_val_cv = X_train_fold[val_split:]
    Y_val_cv = Y_train_fold[val_split:]
    
    # Creo i dataset per train, validation e test
    train_dataset_cv = HARDataset(X_train_cv, Y_train_cv)
    val_dataset_cv = HARDataset(X_val_cv, Y_val_cv)
    test_dataset_cv = HARDataset(X_test_fold, Y_test_fold)
    
    #per ottimizzazione iperparametri 
    def objective_fold(trial):
        lr = trial.suggest_float('lr', 1e-4, 1e-1, log=True)
        batch_size = trial.suggest_categorical('batch_size', [2, 4, 6])
        
        
        train_loader = DataLoader(train_dataset_cv, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset_cv, batch_size=batch_size, shuffle=False, drop_last=True)
        
      
        model = DeepConvLSTM()
        model.load_state_dict(torch.load(r'C:\codes\HumanActivityRecognition\models\best_model_dl_norm_mean_std_and_aug.pkl', 
                                       map_location=torch.device('cpu')), strict=False)
        
        # Sositutusco testa
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, len(unique_labels))
        model.set_n_classes(len(unique_labels))
        
        # Freezo tutti i parametri tranne testa
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True
        
        # Train
        best_f1_score = train_with_cm.train(model, train_loader, val_loader, epochs=100, batch_size=batch_size, lr=lr)
        
        return best_f1_score
    


    
    # Cerco la migliore combinazione lr e batch_size che massimizza F1 sulla validation
    study = optuna.create_study(direction='maximize', pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10))
    study.optimize(objective_fold, n_trials=50)  # Reduced trials per fold
    
    best_params = study.best_params
    logger.info(f"Fold {fold_idx + 1} best params: {best_params}")
    
    # Riaddestro modello con i migliori iperparametri
    best_lr = best_params['lr']
    best_batch_size = best_params['batch_size']
    
    # Create final model for this fold
    final_model = DeepConvLSTM()
    final_model.load_state_dict(torch.load(r'C:\codes\HumanActivityRecognition\models\best_model_dl_norm_mean_std_and_aug.pkl', 
                                          map_location=torch.device('cpu')), strict=False)
    
    num_ftrs = final_model.fc.in_features
    final_model.fc = nn.Linear(num_ftrs, len(unique_labels))
    final_model.set_n_classes(len(unique_labels))
    
    for param in final_model.parameters():
        param.requires_grad = False
    for param in final_model.fc.parameters():
        param.requires_grad = True
    
    # Traino modello finale con i migliori iperparametri
    train_loader_final = DataLoader(train_dataset_cv, batch_size=best_batch_size, shuffle=True, drop_last=True)
    val_loader_final = DataLoader(val_dataset_cv, batch_size=best_batch_size, shuffle=False, drop_last=True)
    
    final_f1 = train_with_cm.train(final_model, train_loader_final, val_loader_final, 
                                  epochs=100, batch_size=best_batch_size, lr=best_lr)
    

    # Salvo il modello finale per questo fold
    model_filename = f"best_model_car_fold_{fold_idx + 1}.pkl"
    model_path = os.path.join(MODELS_DIR, model_filename)
    torch.save(final_model.state_dict(), model_path)
    logger.info(f"Modello del fold {fold_idx + 1} salvato in: {model_path}")

 


    plot_CM(
    mdl_class=DeepConvLSTM,
    mdl_weights=model_path,
    X= X_train_cv,
    Y=Y_train_cv,
    batch_size=best_batch_size,
    figure_name=f"cm_train_fold_{fold_idx + 1}_car_best_hyp_kfold"
    )

    plot_CM(
    mdl_class=DeepConvLSTM,
    mdl_weights=model_path,
    X=X_val_cv,
    Y=Y_val_cv,
    batch_size=best_batch_size,
    figure_name=f"cm_val_fold_{fold_idx + 1}_car_best_hyp_kfold"
    )

    
    

    
    final_model_loaded = DeepConvLSTM(n_classes=len(unique_labels))
    final_model_loaded.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict=False)

    for param in final_model_loaded.parameters():
        param.requires_grad = False  # Congela tutti i pesi


    for name, param in final_model_loaded.named_parameters():
        print(f"{name} requires_grad={param.requires_grad}")

    

    # Valuto sul test set con il modello ricaricato
    test_loader_final = DataLoader(test_dataset_cv, batch_size=best_batch_size, shuffle=False, drop_last=True)
    
    # Valutazione finale con il modello ricaricato
    final_test_loss, final_test_acc, final_test_f1 = train_with_cm.evaluate_model(
        final_model_loaded, test_loader_final, 
        figure_name=f"cm_test_fold_{fold_idx + 1}_car_kfold_final", 
        save_confusion_matrix=True
    )
    
    # Carico risultati del fold
    fold_result = {
        'fold': fold_idx + 1,
        'test_loss': final_test_loss,
        'test_accuracy': final_test_acc,
        'test_f1': final_test_f1,
        'best_lr': best_lr,
        'best_batch_size': best_batch_size,
        'train_size': len(X_train_fold),
        'test_size': len(X_test_fold),
        'train_kids': np.unique(all_kid_ids[train_idx]).tolist(),
        'test_kids': np.unique(all_kid_ids[test_idx]).tolist()
    }
    
    fold_results.append(fold_result)
    all_fold_hyperparams.append(best_params)
    
    logger.info(f"Fold {fold_idx + 1} results: F1={final_test_f1:.4f}, Acc={final_test_acc:.4f}")

# Calcolo le metriche medie e deviazioni standard su tutti i fold
avg_f1 = np.mean([r['test_f1'] for r in fold_results])
avg_acc = np.mean([r['test_accuracy'] for r in fold_results])
std_f1 = np.std([r['test_f1'] for r in fold_results])
std_acc = np.std([r['test_accuracy'] for r in fold_results])

logger.info(f"=== CROSS-VALIDATION RESULTS ===")
logger.info(f"Average F1: {avg_f1:.4f} ± {std_f1:.4f}")
logger.info(f"Average Accuracy: {avg_acc:.4f} ± {std_acc:.4f}")

# Salvo i risultati
results_summary = {
    'cv_method': 'LeaveOneGroupOut' if use_leave_one_out else f'GroupKFold_{n_splits}',
    'average_f1': avg_f1,
    'std_f1': std_f1,
    'average_accuracy': avg_acc,
    'std_accuracy': std_acc,
    'fold_results': fold_results,
    'hyperparameters_per_fold': all_fold_hyperparams,
    'label_mapping': label_mapping
}

# Salvo json
with open(os.path.join(REPORTS_DIR, 'cv_results_car_kfold.json'), 'w') as f:
    json.dump(results_summary, f, indent=2)

# salvo csv
results_df = pd.DataFrame(fold_results)
results_df.to_csv(os.path.join(REPORTS_DIR, 'cv_results_car_kfold.csv'), index=False)

logger.info("Cross-validation completed!")

