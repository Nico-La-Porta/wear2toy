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
from utils.focal_loss import FocalLoss

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
for action_id in df_ball['action_id'].unique():
    # Leggo il file CSV per questa azione
    df_action = pd.read_csv(os.path.join(path, f'df_ball_action_{action_id}.csv'))
    
    logger.info(f"Normalizzando action_id {action_id} - Shape originale: {df_action.shape}")
    
    # Applico la normalizzazione usando la funzione del modulo normalization
    df_action_normalized = normalization.normalize_dataframe_mean_std(df_action, mean, std)
    
    # Salvo il DataFrame normalizzato
    normalized_file_path = os.path.join(path, f'df_ball_action_{action_id}_normalized.csv')
    df_action_normalized.to_csv(normalized_file_path, index=False)
    
    logger.info(f"Salvato file normalizzato: {normalized_file_path}")
    logger.debug(f"Shape del DataFrame normalizzato: {df_action_normalized.shape}")

logger.info("Normalizzazione completata per tutte le azioni")


#applico sliding window con la funzion process_csv
nb_sensor_channels = 9
sliding_window_length = 100
sliding_window_step = 50

#ora applico la funzione sliding window (che mi da come output x_window e y_window) a tutti i .csv relativi al giocattolo ball
#e poi concateno tutto in un unica x e y 

X, Y = [], []
kid_action_counts = {}
normalized_files = [f for f in os.listdir(path) if f.endswith('_normalized.csv') and 'ball_action' in f]
logger.info(f"File normalizzati trovati: {normalized_files}")

for action_file in normalized_files:
    file_path = os.path.join(path, action_file)
    action_id = action_file.split('_')[-2]

    logger.info(f"Processando file normalizzato: {action_file}")

    X_windows, Y_windows, kid_id_action_dict, kid_ids_for_windows = sliding_window_on_data.process_csv(file_path, nb_sensor_channels, sliding_window_length, sliding_window_step)

    X.append(X_windows)
    Y.append(Y_windows)


    logger.info(f"Numero  totale di finestre per l'azione {action_id}:{len(X_windows)}")
    kid_action_counts[f"ball_action_{action_id}"] = kid_id_action_dict #aggiungo il dizionario al dizionario principale per tenere traccia del numero di finestre per ogni bambino per ogni azione, ogni ball_action è una chiave e il valore è un dizionario con il numero di finestre per ogni bambino
    logger.info(f"Contenuto finale di kid_action_counts: {kid_action_counts}") #per veere quante finestre per ogni azione e per ogni bambino sono state elaborte 


# Concateno tutti i dati in un unico array per X e Y
X = np.concatenate(X, axis=0)
Y = np.concatenate(Y, axis=0)


# Stampo le dimensioni di X e Y
logger.info(f"Dimensioni di X finale: {X.shape}")
logger.info(f"Dimensioni di Y finale: {Y.shape}")


logger.info("Pipeline di normalizzazione e sliding window completata!")


from collections import defaultdict
split_ratio_train = 0.7
split_ratio_val = 0.15
split_ratio_test = 0.15

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
        num_train = int(num_windows * split_ratio_train)
        num_test = int(num_windows * split_ratio_test)
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
np.savez(f"{path}\\split_final_indices_ball.npz",
         train=np.array(train_indices_totali),
         val=np.array(val_indices_totali),
         test=np.array(test_indices_totali))

# Stampa controllo
logger.info("Train shape: %s %s", X_train.shape, Y_train.shape)
logger.info("Val shape: %s %s", X_val.shape, Y_val.shape)
logger.info("Test shape: %s %s", X_test.shape, Y_test.shape)


# Ricarico gli indici salvati
loaded_data = np.load(f"{path}\\split_final_indices_ball.npz")

# Estraggo gli indici delle finestre
train_indices_totali = loaded_data['train']
val_indices_totali = loaded_data['val']
test_indices_totali = loaded_data['test']

# Stampa per verificare
logger.info(f"Train indices: {train_indices_totali.shape}")
logger.info(f"Val indices: {val_indices_totali.shape}")
logger.info(f"Test indices: {test_indices_totali.shape}")


# Trova tutte le etichette uniche presenti nei dati
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


# Calcolo i pesi delle classi per la Focal Loss
class_counts = np.bincount(Y_train_mapped) # Conta le occorrenze di ogni classe
total_samples = len(Y_train_mapped) # Totale campioni nel train
class_weights = total_samples / (len(class_counts) * class_counts) # Calcola i pesi delle classi

logger.info(f"Conteggio classi: {class_counts}")
logger.info(f"Pesi delle classi: {class_weights}")


#converto i pesi delle classi in un tensore di PyTorch
class_weights_tensor = torch.FloatTensor(class_weights)
logger.info(f"Class weights tensor: {class_weights_tensor}")

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

# Modifico i nomi dei file per distinguere dal precedente approccio
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model_full_finetuning_ball_norm_mean_std_and_aug.pkl")
BEST_SCORE_PATH = os.path.join(REPORTS_DIR, "best_score_full_finetuning_ball_norm_mean_std_and_aug.txt")

best_hyperparams_file = os.path.join(REPORTS_DIR, 'best_hyperparameters_full_finetuning_ball_norm_mean_std_and_aug.csv')
if os.path.exists(best_hyperparams_file):
    logger.debug(f"Carico i migliori iperparametri da {best_hyperparams_file}")
    best_hyperparameters=pd.read_csv(best_hyperparams_file).iloc[0].to_dict()
    best_lr = best_hyperparameters['lr']
    best_batch_size = int(best_hyperparameters['batch_size'])
    best_gamma = best_hyperparameters.get('gamma', 2.0)  # Default gamma se non presente
    
    # Crea la Focal Loss con i migliori iperparametri
    best_criterion = FocalLoss(
        gamma=best_gamma, 
        alpha=class_weights_tensor, 
        reduction='mean', 
        task_type='multi-class', 
        num_classes=4
    )

else:
    def objective(trial):
        # Definisco gli iperparametri da ottimizzare
        lr = trial.suggest_float('lr', 1e-4, 1e-1, log=True)
        batch_size = trial.suggest_categorical('batch_size', [2,4,6])
        gamma = trial.suggest_float('gamma', 0.5, 3.0)
        # Creo i DataLoader con il batch_size suggerito
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

        # Creo il modello con gli iperparametri suggeriti
        model = DeepConvLSTM(n_classes=4)  # Inizializzo già con 4 classi

        # Carico i pesi del modello pre-trainato (senza strict=False per evitare problemi con la testa)
        pretrained_state_dict = torch.load(r'C:\codes\HumanActivityRecognition\models\best_model_dl_norm_mean_std_and_aug.pkl', map_location=torch.device('cpu'))
        
        # Rimuovo i pesi della testa (fully connected) dal dizionario dei pesi pre-trainati
        # perché il modello pre-trainato potrebbe avere un numero diverso di classi
        pretrained_state_dict = {k: v for k, v in pretrained_state_dict.items() if not k.startswith('fc')}
        
        # Carico i pesi pre-trainati (senza la testa)
        model.load_state_dict(pretrained_state_dict, strict=False)
        
        logger.info("Caricati i pesi pre-trainati per full fine-tuning")

        # NON congelo nessun parametro - tutti i parametri sono addestrabili per il full fine-tuning
        for param in model.parameters():
            param.requires_grad = True
        
        logger.info("Tutti i parametri sono scongelati per full fine-tuning")

        # Creo la Focal Loss con i pesi delle classi
        criterion = FocalLoss(
            gamma=gamma, 
            alpha=class_weights_tensor, 
            reduction='mean', 
            task_type='multi-class', 
            num_classes=4
        )
        # Eseguo l'allenamento con tutti i parametri addestrabili
        best_f1_score = train_with_cm.train(model, train_loader, val_loader, epochs=100, batch_size=batch_size, lr=lr, criterion=criterion)

        # Salvo modello se è il migliore finora
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
            logger.info(f"Nuovo miglior modello salvato in: {BEST_MODEL_PATH} con F1 score: {best_f1_score}")

        return best_f1_score

    # Creazione studio Optuna
    study = optuna.create_study(
        direction='maximize', 
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )
    study.optimize(objective, n_trials=100)

    logger.info(f"Best hyperparameters: {study.best_params}")
    logger.info(f"Highest F1-score: {study.best_value}") 

    # Salvo i best hyperparameters
    best_hyperparameters = study.best_params
    best_hyperparameters['best_f1_score'] = study.best_value
    best_hyperparameters_df = pd.DataFrame([best_hyperparameters])
    best_hyperparameters_df.to_csv(os.path.join(REPORTS_DIR, 'best_hyperparameters_full_finetuning_ball_norm_mean_std_and_aug.csv'), index=False)

    best_lr = study.best_params['lr']
    best_batch_size = study.best_params['batch_size']
    best_gamma = study.best_params['gamma']

    # Visualizzo la storia dell'ottimizzazione
    file_name = "optimization_history_full_finetuning_ball_norm_mean_std_and_aug.png"
    fig = vis.plot_optimization_history(study)
    plt.show()

    # Salvo il grafico
    fig.write_image(os.path.join(FIGURES_DIR, file_name))
    logger.debug(f"Grafico salvato in figures/{file_name}")



# Creo modello finale e carico i migliori pesi
model = DeepConvLSTM(n_classes=4)

# Carico i pesi del miglior modello
model.load_state_dict(
    torch.load(
        BEST_MODEL_PATH,
        map_location=torch.device('cpu')
    )
)




# Verifico che tutti i parametri siano addestrabili
for name, param in model.named_parameters():
    print(f"{name} requires_grad={param.requires_grad}")


# Stampo confusion matrix per il training set
plot_CM(
    mdl_class=DeepConvLSTM,
    mdl_weights=BEST_MODEL_PATH,
    X=X_train,
    Y=Y_train_mapped,
    batch_size=best_batch_size,
    figure_name="cm_train_best_hyp_full_finetuning_ball_norm_mean_std_and_aug"
)

# Stampo confusion matrix per il validation set
plot_CM(
    mdl_class=DeepConvLSTM,
    mdl_weights=BEST_MODEL_PATH,
    X=X_val,
    Y=Y_val_mapped,
    batch_size=best_batch_size,
    figure_name="cm_val_best_hyp_full_finetuning_ball_norm_mean_std_and_aug"
)

# Creo il DataLoader per il test
test_loader = DataLoader(test_dataset, batch_size=best_batch_size, drop_last=True, shuffle=False)

# Eseguo l'evaluation sul test set
test_loss, test_acc, test_f1 = train_with_cm.evaluate_model(
    model, 
    test_loader, 
    figure_name="cm_test_best_hyp_full_finetuning_ball_norm_mean_std_and_aug", 
    save_confusion_matrix=True,
    criterion=best_criterion
)

logger.info(f"Test Results - Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}, F1-Score: {test_f1:.4f}")