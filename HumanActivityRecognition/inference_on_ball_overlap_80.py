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



#applico sliding window con la funzion process_csv
nb_sensor_channels = 9
sliding_window_length = 100
sliding_window_step = 20

#ora applico la funzione sliding window (che mi da come output x_window e y_window) a tutti i .csv relativi al giocattolo ball
#e poi concateno tutto in un unica x e y 

X, Y = [], []
kid_action_counts = {}

for action_file in [f for f in os.listdir(path) if f.endswith('.csv') and f.split('_')[1] == 'ball']:
    file_path = os.path.join(path, action_file)
    action_id = action_file.split('_')[-2]

    logger.info(f"Processando file normalizzato: {action_file}")

    X_windows, Y_windows, kid_id_action_dict, kid_id_windows = sliding_window_on_data.process_csv(file_path, nb_sensor_channels, sliding_window_length, sliding_window_step)

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


X_train_new = X_train.copy()
Y_train_new = Y_train.copy()
X_test_new = X_test.copy()
Y_test_new = Y_test.copy()


indices_to_remove_from_train = []
windows_to_add_to_test = []

azioni_modificate = 0

for action in np.unique(Y):
    train_ids = train_action_counts.get(action, [])
    test_ids = test_action_counts.get(action, [])

    if len(test_ids) == 1 and len(train_ids) >= 3:
        idx_to_move = train_ids[-1]  # prendo l'ultima finestra di quella azione nel train

        logger.debug(f"Azione {action} - Sposto la finestra con indice {idx_to_move} dal train al test")
        logger.debug(f"Y_train[idx]: {Y_train[idx_to_move]}")
        logger.debug(f"X_train[idx][:5]: {X_train[idx_to_move][:5]}")
        logger.debug(f"Shape della finestra: {X_train[idx_to_move].shape}")

        # Salva la finestra da spostare
        windows_to_add_to_test.append({
            'X': X_train[idx_to_move].copy(),
            'Y': Y_train[idx_to_move],
            'original_idx': idx_to_move
        })
        
        # Segna l'indice da rimuovere
        indices_to_remove_from_train.append(idx_to_move)
        
        azioni_modificate += 1
        logger.info(f"Azione {action}: spostata una finestra da train a test per bilanciare meglio")


if indices_to_remove_from_train:
    # Ordino gli indici in ordine decrescente per rimuoverli dalla fine
    indices_to_remove_from_train.sort(reverse=True)
    
    # Rimuovo gli elementi dal train
    X_train_new = np.delete(X_train_new, indices_to_remove_from_train, axis=0)
    Y_train_new = np.delete(Y_train_new, indices_to_remove_from_train, axis=0)
    
    # Aggiunoi gli elementi al test
    for window_data in windows_to_add_to_test:
        X_test_new = np.vstack([X_test_new, window_data['X'].reshape(1, *window_data['X'].shape)])
        Y_test_new = np.append(Y_test_new, window_data['Y'])
        
        logger.debug(f"Aggiunta finestra al test - Y: {window_data['Y']}")
        logger.debug(f"Shape della finestra aggiunta: {window_data['X'].shape}")

# Aggiorno le variabili finali
X_train = X_train_new
Y_train = Y_train_new
X_test = X_test_new
Y_test = Y_test_new

logger.info("Azioni modificate: %d", azioni_modificate)

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

BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model_inference_ball_without_anythinh_overlap_80.pkl")
BEST_SCORE_PATH = os.path.join(REPORTS_DIR, "best_score_inference_ball_without_anythinh_overlap_80.txt")

best_hyperparams_file = os.path.join(REPORTS_DIR, 'best_hyperparameters_inference_inference_ball_without_anythinh_overlap_80.csv')
if os.path.exists(best_hyperparams_file):
    logger.debug(f"Carico i migliori iperparametri da {best_hyperparams_file}")
    best_hyperparameters=pd.read_csv(best_hyperparams_file).iloc[0].to_dict()
    best_lr = best_hyperparameters['lr']
    best_batch_size = int(best_hyperparameters['batch_size'])

else:
    def objective(trial):
        # DefiniscO gli iperparametri da ottimizzare
        lr = trial.suggest_float('lr', 1e-4, 1e-1, log=True)
        batch_size = trial.suggest_categorical('batch_size', [2,4,6])

        # Creo i DataLoader con il batch_size suggerito
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True, drop_last=True)

        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

        # Creo il modello con gli iperparametri suggeriti e rimuovo la testa originale, in modo da non caricare i pesi associati
        model = DeepConvLSTM()

    
        model.load_state_dict(torch.load(r'C:\codes\HumanActivityRecognition\models\best_model_dl_without_anything.pkl', map_location=torch.device('cpu')), strict=False)



        # Ora sostituisco la testa del modello con la nuova dimensione di classi (4)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 4)  # 4 classi
        model.set_n_classes(4)


        # Congelo tutti i parametri tranne quelli della testa (fully connected)
        for param in model.parameters():
            param.requires_grad = False  # Congelo tutti i pesi

        # Sblocco i parametri della testa (fully connected)
        for param in model.fc.parameters():
            param.requires_grad = True  # Solo i pesi della testa saranno addestrabili

        # Eseguo l'allenamento
        best_f1_score = train_with_cm.train(model, train_loader, val_loader, epochs=100, batch_size=batch_size, lr=lr)

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
            torch.save(model.state_dict(), BEST_MODEL_PATH)  # use joblib.dump() if it's not a PyTorch model
            with open(BEST_SCORE_PATH, "w") as f:
                f.write(str(best_f1_score))
            logger.info(f"Nuovo miglior modello salvato in: {BEST_MODEL_PATH} con F1 score: {best_f1_score}")

        return best_f1_score

    # Creazione studio Optuna ottimizza, nel senso di minimizzare la loss in 100 prove
    # Creazione dello studio con il MedianPruner
    study = optuna.create_study(
        direction='maximize', 
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10)  # Parametri di pruning per evitare di continuare trial non promettenti: n_startup_trials=5 significa che i primi 5 trial non verranno prunati, n_warmup_steps=10 significa che dopo 10 trial verrà applicato il pruning
        #Il pruner interromperà automaticamente i trial che non sono promettenti, basandosi sui punteggi parziali (F1-score) ottenuti durante l'allenamento.
    )
    study.optimize(objective, n_trials=100)

    logger.info(f"Best hyperparameters: {study.best_params}")  # Usa f-string
    logger.info(f"Highest F1-score: {study.best_value}") 

    #salvo i best hyperparameters
    best_hyperparameters = study.best_params
    best_hyperparameters['best_f1_score'] = study.best_value
    best_hyperparameters_df = pd.DataFrame([best_hyperparameters])
    best_hyperparameters_df.to_csv(os.path.join(REPORTS_DIR, 'best_hyperparameters_inference_inference_ball_without_anythinh_overlap_80.csv'), index=False)

    best_lr = study.best_params['lr']
    best_batch_size = study.best_params['batch_size']

    #visualizzare la storia dell'ottimizzazione effettuata da Optuna. Ci permette di vedere come l'f1 score
    # è cambiato nel corso delle diverse prove (trials) durante l'ottimizzazione.
    file_name = "optimization_history_inference_ball_without_anythinh_overlap_80.png"
    fig=vis.plot_optimization_history(study)
    plt.show()

    # Salvo il grafico nella cartella FIGURES con il nome specificato
    fig.write_image(os.path.join(FIGURES_DIR, file_name))

    logger.debug(f"Grafico salvato in figures /{file_name}")

#TO DO: ALTRO SCRIPT
#STAMPO CM DELL'ALLENAMENTO SUL TRAINING SET
plot_CM(
    mdl_class=DeepConvLSTM,
    mdl_weights=BEST_MODEL_PATH,
    X= X_train,
    Y=Y_train_mapped,
    batch_size=best_batch_size,
    figure_name="cm_train_best_hyp_inference_optuna_inference_ball_without_anythinh_overlap_80"
)

#STAMPO CM DELL'ALLENAMENTO SUL VAL SET
plot_CM(
    mdl_class=DeepConvLSTM,
    mdl_weights=BEST_MODEL_PATH,
    X=X_val,
    Y=Y_val_mapped,
    batch_size=best_batch_size,
    figure_name="cm_val_best_hyp_inference_optuna_inference_ball_without_anythinh_overlap_80"
)

# Creo modello e carico pesi del miglior modello (trovato prima in optuna)
model = DeepConvLSTM(n_classes=4)

# Rimuovo la testa originale, in modo da non caricare i pesi associati
model.load_state_dict(
    torch.load(
        BEST_MODEL_PATH,
        map_location=torch.device('cpu')
    ),
    strict=False
)


# Congelo tutti i parametri tranne quelli della testa (fully connected)
for param in model.parameters():
        param.requires_grad = False  # Congela tutti i pesi


for name, param in model.named_parameters():
    print(f"{name} requires_grad={param.requires_grad}")



# Creo i DataLoader con i migliori iperparametri
test_loader = DataLoader(test_dataset, batch_size=best_batch_size, drop_last=True, shuffle=False)



# Eseguo l'eval sul test set usando i migliori iperparametri
test_loss, test_acc, test_f1 = train_with_cm.evaluate_model(model, test_loader, figure_name= "cm_test_best_hyp_optuna_inference_ball_without_anythinh_overlap_80", save_confusion_matrix=True)










