import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import glob
import optuna
from optuna.pruners import MedianPruner
import optuna.visualization as vis
from collections import defaultdict

# Import project modules
from src.app_config import FIGURES_DIR
from src.models.DeepConvLSTM import DeepConvLSTM, HARDataset
import src.utils.sliding_window_on_data as sliding_window_on_data
from src.utils import train_with_cm
from src.utils.log_config import logger



# Window parameters for sliding window
NB_SENSOR_CHANNELS = 9
SLIDING_WINDOW_LENGTH = 100
SLIDING_WINDOW_STEP = 50

# Split ratios for train/val/test
SPLIT_RATIO_TRAIN = 0.7
SPLIT_RATIO_VAL = 0.15
SPLIT_RATIO_TEST = 0.15

# Number of classes in the target task
NUM_TARGET_CLASSES = 4

def load_or_create_merged_dataset(data_path):
    """
    Carico il dataset già unito se presente o lo creo unendo i file CSV filtrati con action_id non nullo
    """
    # Nome del file CSV finale
    final_csv_path = os.path.join(data_path, 'df_BA_non_null.csv')

    # Controllo se il file esiste già
    if os.path.exists(final_csv_path):
        logger.debug(f"Il file {final_csv_path} esiste già. Lo sto caricando...")
        df_ball = pd.read_csv(final_csv_path)
    else:
        #trovo tutti i file che corrispondono a "BA" nella cartella path e li stampo a schermo
        files = glob.glob(os.path.join(data_path, "*_BA*.csv"))
        logger.debug("Files:", files)
        
        kid_ball, kid_ball_no_null = [], [] # liste per salvare utenti prima e dopo il merge 
        df_list_ball = [] # lista vuota per appendere i dataframe con attività non nulla

        for file in files:
            df = pd.read_csv(file)
            
            logger.debug("Original shape:", df.shape)
            kid_ball.append(df['kid_id'].unique())
            df = df[df['action_id'] != 0] #filtro le righe con action_id non nullo
            logger.debug("Filtered shape:", df.shape)
            kid_ball_no_null.append(df['kid_id'].unique())

            logger.debug("Columns:", df.columns)
            logger.debug("Action counts:\n", df['action'].value_counts())
            logger.debug("Toy counts:\n", df['toy_id'].value_counts())
            logger.debug("="*50)  # Separatore tra i file

        # mi stampo gli utenti prima di fare il merge e dopo il merge
        logger.info(f"Numero di utenti che hanno fatto almeno un azione: {len(kid_ball_no_null)/len(kid_ball)}")

        '''
        Visto che l'informazione relativa all'id del bambino lo abbiamo, così come abbiamo anche l'informazione relativa
        al giocattolo, vado a filtrare i .csv in modo tale da avere solo le righe che hanno l'attività non nulla e poi
        appendo tutte le righe non nulle in un unico dataframe per tutti i file che terminano in .csv nella balltella path
        '''

        for file in os.listdir(data_path):
            if not file.endswith('.csv'):
                continue
        
            #leggo solo i file che dopo l'undescore ha BA*.csv
            if file.split('_')[-1].startswith('BA') and file.endswith('.csv'): #controllo che il file termini con .csv
                df_temp=pd.read_csv(os.path.join(data_path,file))
                df_temp = df_temp[df_temp['action_id'] != 0]
                df_list_ball.append(df_temp)

        df_ball = pd.concat(df_list_ball)
        logger.info("Dimensioni del df_ball con tutte le attività non nulle")
        logger.info(df_ball.shape)
        logger.info(df_ball.columns)
        logger.info(df_ball['action'].value_counts())

        # salvo il dataframe
        df_ball.to_csv(final_csv_path, index=False)

    return df_ball


def split_dataset_by_action(df_ball, data_path):
    #ora divido il dataframe in base all'attività (action_id) e salvo i dataframe in un file .csv
    #per ogni attività

    for action_id in df_ball['action_id'].unique():
        df_action = df_ball[df_ball["action_id"] == action_id] #filtro il dataframe in base all'attività
        logger.debug(f"Dimensioni del dataframe df_ball_action_{action_id} - {df_action.shape}") #log delle dimensioni del dataframe
        logger.info(f"Conteggio delle attività per df_ball_action_{action_id}") #log del conteggio delle attività
        #salvo il dataframe
        df_action.to_csv(os.path.join(data_path,f'df_ball_action_{action_id}.csv'),index=False) #index=False per non salvare l'indice
        logger.debug(f"Salvato il dataframe df_ball_action_{action_id}.csv")


def apply_sliding_window(data_path):
    """
    Apply sliding window technique to process data
    """
    #ora applico la funzione sliding window (che mi da come output x_window e y_window) a tutti i .csv relativi al giocattolo ball
    #e poi concateno tutto in un unica x e y 

    X, Y = [], []
    kid_action_counts = {}

    for action_file in [f for f in os.listdir(data_path) if f.endswith('.csv') and f.split('_')[1] == 'ball']:
        file_path = os.path.join(data_path, action_file)
        action = action_file.split('_')[-1].split('.')[0]

        X_windows, Y_windows, kid_id_action_dict = sliding_window_on_data.process_csv(file_path,NB_SENSOR_CHANNELS, SLIDING_WINDOW_LENGTH, SLIDING_WINDOW_STEP)

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
    
    return X, Y, kid_action_counts


def create_stratified_split(X, Y, data_path):
    X_train, Y_train = [], []
    X_val, Y_val = [], []
    X_test, Y_test = [], []


    #trovo le azioni uniche
    unique_actions = np.unique(Y)
    logger.info("Azioni uniche: %s", unique_actions)

    #Liste per gli indici delle finestre
    train_indices_totali = []
    val_indices_totali = []
    test_indices_totali = []


    for action in unique_actions:
        # Trovo gli indici delle finestre corrispondenti a ciascuna azione
        action_indices = np.where(Y == action)[0]
        num_windows = len(action_indices)
        logger.info("Azione %s: %d finestre", action, num_windows)
        
        # Gestione casi speciali in base al numero di finestre disponibili
        if num_windows <= 2:
            # Se ci sono solo 1 o 2 finestre, tutte vanno nel training
            train_indices = action_indices
            val_indices = np.array([], dtype=int)
            test_indices = np.array([], dtype=int)
        elif num_windows == 3:
            # Per 3 finestre: 1 train, 1 val, 1 test [1,1,1]
            train_indices = action_indices[:1]
            val_indices = action_indices[1:2]
            test_indices = action_indices[2:]
        elif num_windows == 4:
            # Per 4 finestre: 2 train, 1 val, 1 test [2,1,1]
            train_indices = action_indices[:2]
            val_indices = action_indices[2:3]
            test_indices = action_indices[3:]
        elif num_windows == 5:
            # Per 5 finestre: 3 train, 1 val, 1 test [3,1,1]
            train_indices = action_indices[:3]
            val_indices = action_indices[3:4]
            test_indices = action_indices[4:]
        elif num_windows == 6:
            # Per 6 finestre: 4 train, 1 val, 1 test [4,1,1]
            train_indices = action_indices[:4]
            val_indices = action_indices[4:5]
            test_indices = action_indices[5:]
        else:
            # Per numero di finestre >= 7, uso i rapporti standard
            num_train = int(num_windows * SPLIT_RATIO_TRAIN)
            num_test = int(num_windows * SPLIT_RATIO_TEST)
            num_val = num_windows - num_train - num_test
        
            # Divido gli indici delle finestre in train val e test
            train_indices = action_indices[:num_train]
            val_indices = action_indices[num_train:num_train + num_val]
            test_indices = action_indices[num_train + num_val:]
        

        logger.info("Azione %s - Train: %d, Val: %d, Test: %d", 
                action, len(train_indices), len(val_indices), len(test_indices))

        
        # Aggiungo le finestre al train al val e al test
        X_train.append(X[train_indices])
        Y_train.append(Y[train_indices])
        X_val.append(X[val_indices])
        Y_val.append(Y[val_indices])
        X_test.append(X[test_indices])
        Y_test.append(Y[test_indices])

        # Accumulo gli indici
        train_indices_totali.extend(train_indices.tolist())
        val_indices_totali.extend(val_indices.tolist())
        test_indices_totali.extend(test_indices.tolist())

    # Concateno i dati in un unico array per X e Y
    # Concateno tutti i dati
    X_train = np.concatenate(X_train, axis=0)
    Y_train = np.concatenate(Y_train, axis=0)
    X_val = np.concatenate(X_val, axis=0)
    Y_val = np.concatenate(Y_val, axis=0)
    X_test = np.concatenate(X_test, axis=0)
    Y_test = np.concatenate(Y_test, axis=0)

    # Flatten etichette se necessario
    Y_train = Y_train.flatten()
    Y_val = Y_val.flatten()
    Y_test = Y_test.flatten()

    logger.info("Train shape: %s %s", X_train.shape, Y_train.shape)
    logger.info("Val shape: %s %s", X_val.shape, Y_val.shape)
    logger.info("Test shape: %s %s", X_test.shape, Y_test.shape)

    # Conto quante finestre per ogni azione in train e test
    train_action_counts = defaultdict(list)
    test_action_counts = defaultdict(list)

    for i, label in enumerate(Y_train):
        train_action_counts[label].append(i)

    for i, label in enumerate(Y_test):
        test_action_counts[label].append(i)

    # Liste per nuovi dati aggiornati
    X_train_new, Y_train_new = list(X_train), list(Y_train)
    X_test_new, Y_test_new = list(X_test), list(Y_test)


    azioni_modificate = 0

    for action in np.unique(Y):
        train_ids = train_action_counts.get(action, []) #Indici delle finestre nel train mi servono per vedere se ho almeno 3 finestre
        test_ids = test_action_counts.get(action, []) #Indici delle finestre nel test mi servono per vedere se ho almeno 1 finestra

        if len(test_ids) == 1 and len(train_ids) >= 3:
            idx_to_move = train_ids[-1]  # prendo l'ultima finestra di quella azione nel train

            logger.debug(f"Azione {action} - Sposto la finestra con indice originale {idx_to_move} dal train al test")
            logger.debug(f"Y_train[idx]: {Y_train_new[idx_to_move]}")
            logger.debug(f"X_train[idx][:5]: {X_train_new[idx_to_move][:5]}")  # primi 5 campioni della finestra
            logger.debug(f"Shape della finestra: {X_train_new[idx_to_move].shape}")

            # Aggiungo l'indice della finestra spostata al test
            test_indices_totali.append(train_indices_totali[train_ids[-1]])

            # Rimuovo l'indice dalla lista degli indici di train
            train_indices_totali.remove(train_indices_totali[train_ids[-1]])

            # Sposto la finestra nel test
            X_test_new.append(X_train_new[idx_to_move])
            Y_test_new.append(Y_train_new[idx_to_move])

            #verifico che la finestra sia stata spostata correttamente
            logger.debug(f"Y_test[idx]: {Y_test_new[-1]}")
            logger.debug(f"X_test[idx][:5]: {X_test_new[-1][:5]}")  # primi 5 campioni della finestra
            logger.debug(f"Shape della finestra: {X_test_new[-1].shape}")

            # La rimuovo dal train
            del X_train_new[idx_to_move]
            del Y_train_new[idx_to_move]

            azioni_modificate += 1
            logger.info(f"Azione {action}: spostata una finestra da train a test per bilanciare meglio")

    # Converto di nuovo in numpy
    X_train = np.array(X_train_new)
    Y_train = np.array(Y_train_new)
    X_test = np.array(X_test_new)
    Y_test = np.array(Y_test_new)

    logger.info("Azioni modificate: %d", azioni_modificate)


    # Salvo gli indici finali dopo il bilanciamento
    np.savez(f"{data_path}\\split_final_indices_ball.npz",
            train=np.array(train_indices_totali),
            val=np.array(val_indices_totali),
            test=np.array(test_indices_totali))

    # Stampa controllo
    logger.info("Train shape: %s %s", X_train.shape, Y_train.shape)
    logger.info("Val shape: %s %s", X_val.shape, Y_val.shape)
    logger.info("Test shape: %s %s", X_test.shape, Y_test.shape)
    
    return X_train, Y_train, X_val, Y_val, X_test, Y_test


def create_label_mapping(Y_train, Y_val, Y_test):
    unique_labels = np.unique(Y_train)

    # Crea un dizionario che mappa ogni etichetta originale a un valore consecutivo
    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}

    # Stampa il dizionario per vedere il mapping
    logger.info("Mapping delle etichette: %s", label_mapping)

    # Applica il mapping ai dataset di train e test
    Y_train_mapped = np.array([label_mapping[y] for y in Y_train])
    Y_test_mapped = np.array([label_mapping[y] for y in Y_test])
    Y_val_mapped = np.array([label_mapping[y] for y in Y_val])

    # Controllo finale
    logger.info("Nuove etichette train: %s", np.unique(Y_train_mapped))
    logger.info("Nuove etichette val: %s ", np.unique(Y_val_mapped))
    logger.info("Nuove etichette test: %s", np.unique(Y_test_mapped))
    
    return Y_train_mapped, Y_val_mapped, Y_test_mapped


def create_datasets_and_loaders(X_train, Y_train_mapped, X_val, Y_val_mapped, X_test, Y_test_mapped, batch_size=None):
    # Creo i dataset per il training val e test
    train_dataset = HARDataset(X_train, Y_train_mapped)
    val_dataset = HARDataset(X_val, Y_val_mapped)
    test_dataset = HARDataset(X_test, Y_test_mapped)

    #stampo il dataset di training e di test a livello di dimensioni
    logger.info("Lunghezza dataset di training: %s", len(train_dataset))
    logger.info("Lunghezza dataset di validazione: %s", len(val_dataset))
    logger.info("Lunghezza dataset di test: %s", len(test_dataset))


    logger.debug(f"Tipo di train_dataset: {type(train_dataset)}")
    logger.debug(f"Tipo di val_dataset: {type(val_dataset)}")
    logger.debug(f"Tipo di test_dataset: {type(test_dataset)}")
    
    # Create data loaders if batch_size is provided
    if batch_size is not None:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
        return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader
    
    return train_dataset, val_dataset, test_dataset


def create_transfer_learning_model(pretrained_path, num_classes):
    """
    Create a transfer learning model:
    1. Load pretrained model
    2. Replace classification head
    3. Freeze all layers except the new head
    """
    # Create base model
    model = DeepConvLSTM()
    
    # Load pretrained weights (exclude final layer)
    model.load_state_dict(
        torch.load(pretrained_path, map_location=torch.device('cpu')), 
        strict=False
    )
    
    # Replace classification head with new one for target task
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    model.set_n_classes(num_classes)
    
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze only the classification head
    for param in model.fc.parameters():
        param.requires_grad = True
    
    # Verify parameter freezing
    for name, param in model.named_parameters():
        logger.debug(f"{name}: requires_grad={param.requires_grad}")
    
    return model


def optimize_hyperparameters(train_dataset, val_dataset, n_trials=100):
    """
    Optimize hyperparameters using Optuna
    """
    def objective(trial):
        """Optuna objective function"""
        # Define hyperparameters to optimize
        lr = trial.suggest_float('lr', 1e-4, 1e-1, log=True)
        batch_size = trial.suggest_categorical('batch_size', [2, 4, 6])
        
        # Create data loaders with suggested batch size
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
        
        # Create transfer learning model
        model = create_transfer_learning_model(PRETRAINED_MODEL_PATH, NUM_TARGET_CLASSES)
        
        # Train model with suggested hyperparameters
        best_f1_score = train_with_cm.train(
            model, train_loader, val_loader, 
            epochs=100, batch_size=batch_size, lr=lr
        )
        
        # Save model if it's the best so far
        is_better = False
        if not os.path.exists(BEST_SCORE_PATH):
            is_better = True
        else:
            with open(BEST_SCORE_PATH, "r") as f:
                best_score_so_far = float(f.read())
            if best_f1_score > best_score_so_far:
                is_better = True
        
        if is_better:
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            with open(BEST_SCORE_PATH, "w") as f:
                f.write(str(best_f1_score))
            logger.info(f"New best model saved at {BEST_MODEL_PATH} with F1 score: {best_f1_score}")
        
        return best_f1_score
    
    # Create Optuna study with median pruner
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    study = optuna.create_study(direction='maximize', pruner=pruner)
    
    # Run optimization
    study.optimize(objective, n_trials=n_trials)
    
    # Log results
    logger.info(f"Best hyperparameters: {study.best_params}")
    logger.info(f"Highest F1-score: {study.best_value}")
    
    # Save best hyperparameters
    best_hyperparameters = study.best_params
    best_hyperparameters['best_f1_score'] = study.best_value
    best_hyperparameters_df = pd.DataFrame([best_hyperparameters])
    best_hyperparameters_df.to_csv(BEST_HYPERPARAMS_FILE, index=False)
    
    # Save optimization history plot
    fig = vis.plot_optimization_history(study)
    fig.write_image(os.path.join(FIGURES_DIR, "optimization_inference_ball_history_dl_without_anything.png"))
    
    return study.best_params


def evaluate_final_model(test_dataset, best_hyperparams):
    """
    Evaluate final model on train, validation and test sets
    """
    # Create data loaders with best batch size
    best_batch_size = int(best_hyperparams['batch_size'])
    test_loader = DataLoader(test_dataset, batch_size=best_batch_size, shuffle=False, drop_last=True)
    
    # Load best model
    model = DeepConvLSTM(n_classes=NUM_TARGET_CLASSES)
    model.load_state_dict(
        torch.load(BEST_MODEL_PATH, map_location=torch.device('cpu')),
        strict=False
    )
    
    # Freeze all parameters for evaluation
    for param in model.parameters():
        param.requires_grad = False
    
    
    # Evaluate model on test set
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, test_f1 = train_with_cm.evaluate_model(
        model, 
        test_loader, 
        figure_name="cm_test_best_hyp_optuna_dl_without_norm_without_sampler_aug_4_classes", 
        save_confusion_matrix=True
    )
    
    logger.info(f"Test results - Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}, F1 Score: {test_f1:.4f}")
    
    return test_loss, test_acc, test_f1