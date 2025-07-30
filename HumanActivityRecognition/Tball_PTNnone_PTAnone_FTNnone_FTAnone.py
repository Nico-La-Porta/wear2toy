import os
import pandas as pd
import numpy as np
from app_config import REPORTS_DIR, FIGURES_DIR, MODELS_DIR
import sliding_window_on_data
from torch.utils.data import DataLoader
import glob

from models.DeepConvLSTM import DeepConvLSTM, HARDataset 
import optuna
from optuna.pruners import MedianPruner
import optuna.visualization as vis
import train
import torch
import torch.nn as nn
import train_with_cm
import matplotlib.pyplot as plt
from collections import defaultdict
import logging
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
#definisco il path da cui leggere i .csv

from utils.log_config import logger
from figures import plot_CM, combine_kfold_confusion_matrices
from utils import mapping_activity
from HumanActivityRecognition.results_analysis import *

import normalization
from sklearn.model_selection import StratifiedKFold

#definisco il path da cui leggere i .csv

path="C:\codes\HumanActivityRecognition\data\downstream_data"
logger.debug(path)



#Visto che l'informazione relativa all'id del bambino lo abbiamo, così come abbiamo anche l'informazione relativa
#al giocattolo, vado a filtrare i .csv in modo tale da avere solo le righe che hanno l'attività non nulla e poi
#appendo tutte le righe delle righe non nulle in un unico dataframe
#per tutti i file che terminano in .csv nella cartella path
# Definisco il percorso della cartella contenente i CSV

# Nome del file CSV finale
final_csv_path = os.path.join(path, 'df_BA_non_null.csv')

# Controllo se il file esiste già
if os.path.exists(final_csv_path):
    logger.debug(f"Il file {final_csv_path} esiste già. Lo sto caricando...")
    df_ball = pd.read_csv(final_csv_path)
else:
    #trovo tutti i file che corrispondono a "BA" nella cartella path e li stampo a schermo
    files = glob.glob(os.path.join(path, "*_BA*.csv"))
    logger.debug(f"Files: {files}")
    
    kid_ball, kid_ball_no_null = [], [] # liste per salvare utenti prima e dopo il merge 
    df_list_ball = [] # lista vuota per appendere i dataframe con attività non nulla

    for file in files:
        df = pd.read_csv(file)
        
        logger.debug(f"Original shape: {df.shape}")
        kid_ball.append(df['kid_id'].unique())
        df = df[df['action_id'] != 0] #filtro le righe con action_id non nullo
        logger.debug(f"Filtered shape: {df.shape}")
        kid_ball_no_null.append(df['kid_id'].unique())

        logger.debug(f"Columns: {df.columns}")
        logger.debug(f"Action counts:\n{df['action'].value_counts()}")
        logger.debug(f"Toy counts:\n{df['toy_id'].value_counts()}")
        logger.debug("="*50)  # Separatore tra i file

    # mi stampo gli utenti prima di fare il merge e dopo il merge
    if len(kid_ball) > 0:
        logger.info(f"Numero di utenti che hanno fatto almeno un azione: {len(kid_ball_no_null)/len(kid_ball)}")
    else:
        logger.info("Nessun utente trovato nei file analizzati (kid_ball è vuoto).")

    '''
    Visto che l'informazione relativa all'id del bambino lo abbiamo, così come abbiamo anche l'informazione relativa
    al giocattolo, vado a filtrare i .csv in modo tale da avere solo le righe che hanno l'attività non nulla e poi
    appendo tutte le righe non nulle in un unico dataframe per tutti i file che terminano in .csv nella balltella path
    '''

    for file in os.listdir(path):
        if not file.endswith('.csv'):
            continue
    
        #leggo solo i file che dopo l'undescore ha BA*.csv
        if file.split('_')[-1].startswith('BA') and file.endswith('.csv'): #controllo che il file termini con .csv
            df_temp=pd.read_csv(os.path.join(path,file))
            df_temp = df_temp[df_temp['action_id'] != 0]
            df_list_ball.append(df_temp)

    df_ball = pd.concat(df_list_ball)
    logger.info("Dimensioni del df_ball con tutte le attività non nulle")
    logger.info(df_ball.shape)
    logger.info(df_ball.columns)
    logger.info(df_ball['action'].value_counts())

    # salvo il dataframe
    df_ball.to_csv(os.path.join(path,'df_BA_non_null.csv'),index=False)


#ora divido il dataframe in base all'attività (action_id) e salvo i dataframe in un file .csv
#per ogni attività

for action_id in df_ball['action_id'].unique():
    df_action = df_ball[df_ball["action_id"] == action_id] #filtro il dataframe in base all'attività
    logger.debug(f"Dimensioni del dataframe df_ball_action_{action_id} - {df_action.shape}") #log delle dimensioni del dataframe
    logger.info(f"Conteggio delle attività per df_ball_action_{action_id}") #log del conteggio delle attività
    #salvo il dataframe
    df_action.to_csv(os.path.join(path,f'df_ball_action_{action_id}.csv'),index=False) #index=False per non salvare l'indice
    logger.debug(f"Salvato il dataframe df_ball_action_{action_id}.csv")


#applico sliding window con la funzion process_csv
nb_sensor_channels = 9
sliding_window_length = 100
sliding_window_step = 50

#ora applico la funzione sliding window (che mi da come output x_window e y_window) a tutti i .csv relativi al giocattolo ball
#e poi concateno tutto in un unica x e y 

X, Y = [], []
kid_action_counts = {}

for action_file in [f for f in os.listdir(path) if f.endswith('.csv') and f.startswith('df_ball_action_') and not f.endswith('normalized_pt.csv')]:
    file_path = os.path.join(path, action_file)
    action = action_file.split('_')[-1].split('.')[0]

    X_windows, Y_windows, kid_id_action_dict = sliding_window_on_data.process_csv(file_path, nb_sensor_channels, sliding_window_length, sliding_window_step)

    X.append(X_windows)
    Y.append(Y_windows)


    logger.info(f"Numero  totale di finestre per l'azione {action}:{len(X_windows)}")
    kid_action_counts[f"ball_action_{action}"] = kid_id_action_dict #aggiungo il dizionario al dizionario principale per tenere traccia del numero di finestre per ogni bambino per ogni azione, ogni ball_action è una chiave e il valore è un dizionario con il numero di finestre per ogni bambino
    logger.info(f"Contenuto finale di kid_action_counts: {kid_action_counts}") #per veere quante finestre per ogni azione e per ogni bambino sono state elaborte 


# Concateno tutti i dati in un unico array per X e Y
X = np.concatenate(X, axis=0)
Y = np.concatenate(Y, axis=0)


# Stampo le dimensioni di X e Y
logger.info(f"Dimensioni di X finale: {X.shape}")
logger.info(f"Dimensioni di Y finale: {Y.shape}")

logger.debug(f"Tipo di Y: {type(Y)}")
logger.debug(f"Tipo del primo elemento di Y: {type(Y[0])}")
logger.debug(f"Shape del primo elemento di Y: {Y[0].shape if hasattr(Y[0], 'shape') else 'No shape'}")
logger.debug(f"Primi 5 elementi di Y: {Y[:5]}")


# Trova tutte le etichette uniche e crea mapping
unique_labels = np.unique(Y)
label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
logger.info("Mapping delle etichette: %s", label_mapping)
Y_flat = Y.flatten()  # Converte da (28, 1) a (28,)
# Applica il mapping
Y_mapped = np.array([label_mapping[y] for y in Y_flat])
logger.info("Nuove etichette: %s", np.unique(Y_mapped))


# Usa direttamente il mapping hardcodato
ball_mapping = mapping_activity.BALL_ACTION_MAPPING
print(f"Ball mapping keys: {list(ball_mapping.keys())}")
labels_dict = ball_mapping["encoded_to_name"]
print(f"Labels dict: {labels_dict}")


class_names = [ball_mapping["encoded_to_name"][i] for i in range(4)]
print(f"Class names: {class_names}")

# Parametri per K-Fold
K_FOLDS = 3
random_state = 42

# Creo il StratifiedKFold
skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=random_state)

# Dizionario per salvare i risultati di ogni fold
fold_results = {
    'fold': [],
    'best_f1_score': [],
    'best_params': [],
    'test_f1_score': []
}

# Percorsi per i modelli
BEST_MODEL_KFOLD_DIR = os.path.join(MODELS_DIR, "kfold_ball_models_3_folds")
os.makedirs(BEST_MODEL_KFOLD_DIR, exist_ok=True)


logger.info("=== INIZIO STRATIFIED K-FOLD CROSS-VALIDATION ===")
logger.info(f"Numero di fold: {K_FOLDS}")


# Lista per salvare tutti i punteggi per la selezione finale
all_fold_scores = []


for fold, (train_val_idx, test_idx) in enumerate(skf.split(X, Y_mapped)):
    logger.info(f"\n=== FOLD {fold + 1}/{K_FOLDS} ===")
    
    # Split dei dati per questo fold
    X_train_val, X_test_fold = X[train_val_idx], X[test_idx]
    Y_train_val, Y_test_fold = Y_mapped[train_val_idx], Y_mapped[test_idx]
    
    # Ulteriore split di train_val in train e validation (80-20)
    train_val_split = int(0.8 * len(X_train_val))
    
    X_train_fold = X_train_val[:train_val_split]
    Y_train_fold = Y_train_val[:train_val_split]
    X_val_fold = X_train_val[train_val_split:]
    Y_val_fold = Y_train_val[train_val_split:]
    
    logger.info(f"Fold {fold + 1} - Train: {X_train_fold.shape}, Val: {X_val_fold.shape}, Test: {X_test_fold.shape}")
    logger.info(f"Fold {fold + 1} - Distribuzione test: {np.bincount(Y_test_fold)}")

    # Creo i dataset per questo fold
    train_dataset_fold = HARDataset(X_train_fold, Y_train_fold)
    val_dataset_fold = HARDataset(X_val_fold, Y_val_fold)
    test_dataset_fold = HARDataset(X_test_fold, Y_test_fold)


    # Ottimizzazione degli iperparametri per questo fold
    def objective_fold(trial):
        lr = trial.suggest_float('lr', 1e-4, 1e-1, log=True)
        batch_size = trial.suggest_categorical('batch_size', [2, 4])
        
        # Creo i DataLoader
        train_loader = DataLoader(train_dataset_fold, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset_fold, batch_size=batch_size, shuffle=False, drop_last=True)
        
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
        model.classification_head = nn.Linear(num_ftrs, 4)
        model.set_n_classes(4)

        for param in model.parameters():
            param.requires_grad = True
        
        # Training
        best_f1_score = train_with_cm.train(model, train_loader, val_loader, 
                                           epochs=100, batch_size=batch_size, lr=lr)
        
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
    model_final_fold.classification_head = nn.Linear(num_ftrs, 4)
    model_final_fold.set_n_classes(4)

    for param in model_final_fold.parameters():
        param.requires_grad = True

    # Training finale per questo fold
    train_loader_final = DataLoader(train_dataset_fold, batch_size=best_batch_size, shuffle=True, drop_last=True)
    val_loader_final = DataLoader(val_dataset_fold, batch_size=best_batch_size, shuffle=False, drop_last=True)
    
    final_f1_score = train_with_cm.train(model_final_fold, train_loader_final, val_loader_final,
                                        epochs=100, batch_size=best_batch_size, lr=best_lr)
    

    # Valutazione sul test set di questo fold
    test_loader_fold = DataLoader(test_dataset_fold, batch_size=best_batch_size, shuffle=False, drop_last=True)
    test_loss, test_acc, test_f1_fold = train_with_cm.evaluate_model(
        model_final_fold, test_loader_fold, 
        figure_name=f"cm_test_fold_{fold + 1}_kfold_inference_ball_3_folds_Tball_PTNnone_PTAnone_FTNnone_FTAnone",
        save_confusion_matrix=True, 
        save_predictions_csv=True,
        save_f1_score=True,
        labels_dict=labels_dict
    )

    # Salvo il modello di questo fold
    model_path_fold = os.path.join(BEST_MODEL_KFOLD_DIR, f"best_model_fold_{fold + 1}_without_anything.pkl")
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
results_df.to_csv(os.path.join(REPORTS_DIR, 'kfold_results_inference_ball_3_folds_Tball_PTNnone_PTAnone_FTNnone_FTAnone.csv'), index=False)

# Salvo i migliori iperparametri
best_hyperparameters = best_fold_result['params']
best_hyperparameters['mean_test_f1'] = mean_f1
best_hyperparameters['std_test_f1'] = std_f1
best_hyperparameters['best_fold'] = best_fold_result['fold']

best_hyperparameters_df = pd.DataFrame([best_hyperparameters])
best_hyperparameters_df.to_csv(
    os.path.join(REPORTS_DIR, 'best_hyperparameters_kfold_inference_ball_3_folds_Tball_PTNnone_PTAnone_FTNnone_FTAnone.csv'), 
    index=False
)

# === PARTE 5: TRAINING FINALE SU TUTTI I DATI ===

logger.info("\n=== TRAINING FINALE SU TUTTI I DATI ===")

# Uso i migliori iperparametri per il training finale
best_lr_final = best_hyperparameters['lr']
best_batch_size_final = best_hyperparameters['batch_size']

# Creo dataset con tutti i dati
full_dataset = HARDataset(X, Y_mapped)

# Split finale 85-15 per train-val
train_size = int(0.85 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_dataset_final, val_dataset_final = torch.utils.data.random_split(
    full_dataset, [train_size, val_size], 
    generator=torch.Generator().manual_seed(random_state)
)

# Creo DataLoader finali
train_loader_final = DataLoader(train_dataset_final, batch_size=best_batch_size_final, shuffle=True, drop_last=True)
val_loader_final = DataLoader(val_dataset_final, batch_size=best_batch_size_final, shuffle=False, drop_last=True)

# Creo modello finale
model_final = DeepConvLSTM()
model_final.load_state_dict(
    torch.load(r'C:\codes\HumanActivityRecognition\models\best_model_dl_without_anything.pkl', 
              map_location=torch.device('cpu')), 
    strict=False
)

# Sostituisco la testa
num_ftrs = model_final.classification_head.in_features
model_final.classification_head = nn.Linear(num_ftrs, 4)
model_final.set_n_classes(4)

for param in model_final.parameters():
    param.requires_grad = True

# Training finale
logger.info("Inizio training finale su tutti i dati...")
final_f1_score = train_with_cm.train(
    model_final, train_loader_final, val_loader_final,
    epochs=100, batch_size=best_batch_size_final, lr=best_lr_final
)

# Salvo il modello finale
FINAL_MODEL_PATH = os.path.join(MODELS_DIR, "best_model_kfold_final_inference_ball_3_folds_Tball_PTNnone_PTAnone_FTNnone_FTAnone.pkl")
torch.save(model_final.state_dict(), FINAL_MODEL_PATH)

logger.info(f"Modello finale salvato: {FINAL_MODEL_PATH}")
logger.info(f"F1 score finale: {final_f1_score:.4f}")

# === PARTE 6: VALUTAZIONE FINALE ===

# Confusion matrix su tutti i dati
plot_CM(
    mdl_class=DeepConvLSTM,
    mdl_weights=FINAL_MODEL_PATH,
    X=X,
    Y=Y_mapped,
    batch_size=best_batch_size_final,
    figure_name="cm_final_kfold_inference_ball_all_data_3_folds_Tball_PTNnone_PTAnone_FTNnone_FTAnone",
    labels_dict=labels_dict
)

logger.info("=== PROCEDURA K-FOLD COMPLETATA ===")
logger.info(f"Migliori iperparametri: {best_hyperparameters}")
logger.info(f"F1 score medio K-Fold: {mean_f1:.4f} ± {std_f1:.4f}")
logger.info(f"F1 score finale: {final_f1_score:.4f}")





# Per ball
# Per ball - VERSIONE AUTOMATICA
cm_combined, y_true, y_pred = combine_kfold_confusion_matrices(
    fold_results_dir=REPORTS_DIR,
    num_folds=3,
    toy_name="ball",
    save_path="combined_cm_ball_3fold_Tball_PTNnone_PTAnone_FTNnone_FTAnone.png",
    class_names=class_names
    # script_suffix viene dedotto automaticamente dal nome del file
)

#AGGREGAZIONE DELLE CM DI TEST
cm_combined, y_true, y_pred = combine_kfold_confusion_matrices(
    fold_results_dir=REPORTS_DIR,
    num_folds=3,
    toy_name="ball",
    save_path="combined_cm_ball_3fold_Tball_PTNnone_PTAnone_FTNnone_FTAnone.png",
    class_names=class_names
)

"""# === ANALISI APPROFONDITA DEGLI ERRORI ===
logger.info("=== INIZIO ANALISI APPROFONDITA DEGLI ERRORI ===")

# 1. Analisi temporale degli errori
temporal_results = analyze_temporal_error_distribution(
    y_true=y_true,
    y_pred=y_pred
)

plot_temporal_error_distribution(
    results=temporal_results,
    class_names=class_names,
    save_path=os.path.join(FIGURES_DIR, "temporal_error_analysis_ball.png")
)

# 2. Analisi errori con padding
# Per questo hai bisogno dei dati X originali dai fold
# Se non li hai salvati, puoi ricostruirli o usare i dati completi come approssimazione
padding_results = analyze_padding_errors(
    y_true=y_true,
    y_pred=y_pred,
    X_data=X,  # Usa i dati completi come approssimazione
    padding_threshold=0.01,  # Regola in base ai tuoi dati
    save_path=os.path.join(FIGURES_DIR, "padding_error_analysis_ball.png")
)

# 3. Analisi errori per soggetto
# Per questo hai bisogno degli ID dei soggetti per ogni finestra
# Se li hai salvati durante il k-fold, caricali, altrimenti usa una stima
if 'kid_action_counts' in locals():
    # Ricostruisci gli ID dei soggetti (se possibile)
    logger.info("Ricostruzione degli ID soggetti in corso...")
    # Questo è un esempio - adatta in base ai tuoi dati
    subject_ids = np.random.randint(3000, 3025, len(y_true))  # Placeholder
    
    subject_results, worst_subjects, best_subjects = analyze_subject_errors(
        y_true=y_true,
        y_pred=y_pred,
        subject_ids=subject_ids,
        class_names=class_names,
        save_path=os.path.join(FIGURES_DIR, "subject_error_analysis_ball.png")
    )
    
    logger.info(f"Soggetti più problematici: {worst_subjects}")
    logger.info(f"Soggetti con migliori performance: {best_subjects}")

logger.info("=== ANALISI ERRORI COMPLETATA ===")"""