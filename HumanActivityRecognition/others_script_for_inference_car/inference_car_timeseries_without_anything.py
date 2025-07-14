import os
import pandas as pd
import numpy as np
from app_config import REPORTS_DIR, FIGURES_DIR, MODELS_DIR
import sliding_window_on_data
from torch.utils.data import DataLoader
import glob
from collections import defaultdict

from models.DeepConvLSTM import DeepConvLSTM, HARDataset 
import optuna
import optuna.visualization as vis
import train
import torch
import train_with_cm
import matplotlib.pyplot as plt
#definisco il path da cui leggere i .csv

from utils.log_config import logger
from utils.data_preprocessing import remove_classes_from_car_data, get_class_distribution

from optuna.pruners import MedianPruner

from figures import plot_CM
import torch.nn as nn

#definisco il path da cui leggere i .csv

path='C:\codes\HumanActivityRecognition\data\downstream_data'
print(path)

#Visto che l'informazione relativa all'id del bambino lo abbiamo, così come abbiamo anche l'informazione relativa
#al giocattolo, vado a filtrare i .csv in modo tale da avere solo le righe che hanno l'attività non nulla e poi
#appendo tutte le righe delle righe non nulle in un unico dataframe
#per tutti i file che terminano in .csv nella cartella path
# Definisco il percorso della cartella contenente i CSV

# Nome del file CSV finale
final_csv_path = os.path.join(path, 'df_CAR_non_null.csv')

# Controllo se il file esiste già
if os.path.exists(final_csv_path):
    print(f"Il file {final_csv_path} esiste già. Lo sto caricando...")
    df_car = pd.read_csv(final_csv_path)
else:
    #trovo tutti i file che corrispondono a "C*" nella cartella path e li stampo a schermo
    files = glob.glob(os.path.join(path, "*_C*.csv"))
    print("Files:", files)
    
    kid_car, kid_car_no_null = [], [] # liste per salvare utenti prima e dopo il merge 
    df_list_car = [] # lista vuota per appendere i dataframe con attività non nulla

    for file in files:
        df = pd.read_csv(file)
        
        print("Original shape:", df.shape)
        kid_car.append(df['kid_id'].unique())
        df = df[df['action_id'] != 0] #filtro le righe con action_id non nullo
        print("Filtered shape:", df.shape)
        kid_car_no_null.append(df['kid_id'].unique())

        print("Columns:", df.columns)
        print("Action counts:\n", df['action'].value_counts())
        print("Toy counts:\n", df['toy_id'].value_counts())
        print("="*50)  # Separatore tra i file

    # mi stampo gli utenti prima di fare il merge e dopo il merge
    logger.info(f"Numero di utenti che hanno fatto almeno un azione: {len(kid_car_no_null)/len(kid_car)}")

    '''
    Visto che l'informazione relativa all'id del bambino lo abbiamo, così come abbiamo anche l'informazione relativa
    al giocattolo, vado a filtrare i .csv in modo tale da avere solo le righe che hanno l'attività non nulla e poi
    appendo tutte le righe non nulle in un unico dataframe per tutti i file che terminano in .csv nella cartella path
    '''

    for file in os.listdir(path):
        if not file.endswith('.csv'):
            continue
    
        #leggo solo i file che dopo l'undescore ha BE*.csv
        if file.split('_')[-1].startswith('C') and file.endswith('.csv'): #controllo che il file termini con .csv
            df_temp=pd.read_csv(os.path.join(path,file))
            df_temp = df_temp[df_temp['action_id'] != 0]
            df_list_car.append(df_temp)

    df_car = pd.concat(df_list_car)
    print("Dimensioni del df_car con tutte le attività non nulle")
    print(df_car.shape)
    print(df_car.columns)
    print(df_car['action'].value_counts())

    # salvo il dataframe
    df_car.to_csv(os.path.join(path,'df_CAR_non_null.csv'),index=False)

#filtro le classi per trovare solo quelle che mi interessano
logger.info("FILTRAGGIO CLASSI DAL DATASET CAR")

# Mostra distribuzione originale
logger.info("Distribuzione action_id originale:")
original_counts = df_car['action_id'].value_counts().sort_index()
for action_id, count in original_counts.items():
    logger.info(f"   Action {action_id}: {count} righe")


# Rimuovo le classi che non ti interessano
classes_to_remove = [2, 3, 5, 9, 11, 12, 14, 16, 18, 19, 27, 28, 29, 31, 32, 37, 38, 39, 40]
logger.info(f"Rimozione action_id: {classes_to_remove}")

df_car_filtered = df_car[~df_car['action_id'].isin(classes_to_remove)] # filtro il dataframe per rimuovere le classi che non mi interessano

logger.info(f"Righe prima del filtro: {len(df_car)}")
logger.info(f"Righe dopo il filtro: {len(df_car_filtered)}")
logger.info(f"Righe rimosse: {len(df_car) - len(df_car_filtered)}")


logger.info("Distribuzione action_id dopo filtro:")
filtered_counts = df_car_filtered['action_id'].value_counts().sort_index()
for action_id, count in filtered_counts.items():
    logger.info(f"   Action {action_id}: {count} righe")


#ora divido il dataframe in base all'attività (action_id) e salvo i dataframe in un file .csv
#per ogni attività

for action_id in df_car_filtered['action_id'].unique():
    df_action = df_car_filtered[df_car_filtered["action_id"] == action_id] #filtro il dataframe in base all'attività
    logger.debug(f"Dimensioni del dataframe df_car_action_{action_id} - {df_action.shape}") #log delle dimensioni del dataframe
    logger.info(f"Conteggio delle attività per df_car_action_{action_id}") #log del conteggio delle attività
    #salvo il dataframe
    df_action.to_csv(os.path.join(path,f'df_car_action_{action_id}.csv'),index=False) #index=False per non salvare l'indice
    logger.debug(f"Salvato il dataframe df_car_action_{action_id}.csv")

#applico sliding window con la funzion process_csv
nb_sensor_channels = 9
sliding_window_length = 100
sliding_window_step = 50

#ora applico la funzione sliding window (che mi da come output x_window e y_window) a tutti i .csv relativi al giocattolo car
#e poi concateno tutto in un unica x e y 

X, Y = [], []
kid_action_counts = {}


for action_file in [f for f in os.listdir(path) if f.endswith('.csv') and f.split('_')[1] == 'car']:
    file_path = os.path.join(path, action_file)
    action = action_file.split('_')[-1].split('.')[0]

    X_windows, Y_windows, kid_id_action_dict = sliding_window_on_data.process_csv(file_path, nb_sensor_channels, sliding_window_length, sliding_window_step)

    X.append(X_windows)
    Y.append(Y_windows)


    logger.info(f"Numero  totale di finestre per l'azione {action}:{len(X_windows)}")
    kid_action_counts[f"car_action_{action}"] = kid_id_action_dict #aggiungo il dizionario al dizionario principale per tenere traccia del numero di finestre per ogni bambino per ogni azione, ogni car_action è una chiave e il valore è un dizionario con il numero di finestre per ogni bambino
    logger.info(f"Contenuto finale di kid_action_counts: {kid_action_counts[f'car_action_{action}']}") #per vedere quante finestre per ogni azione e per ogni bambino sono state elaborate


# Concateno tutti i dati in un unico array per X e Y
X = np.concatenate(X, axis=0)
Y = np.concatenate(Y, axis=0)


# Stampo le dimensioni di X e Y
logger.info(f"Dimensioni di X finale: {X.shape}")
logger.info(f"Dimensioni di Y finale: {Y.shape}")
logger.info(f"Tipo di Y: {Y.dtype}")
logger.info(f"Shape di Y: {Y.shape}")
logger.info(f"Primi 5 elementi di Y: {Y[:5]}")
logger.info(f"Tipo del primo elemento: {type(Y[0])}")


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
get_class_distribution(Y_remapped, "Car Dataset con Etichette Rimappate")

#uso le Y_remapped per il resto del codice
Y = Y_remapped


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
        #logger.debug(f"X_train[idx][:5]: {X_train_new[idx_to_move][:5]}")  # primi 5 campioni della finestra
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
np.savez(f"{path}\\split_final_indices_car.npz",
         train=np.array(train_indices_totali),
         val=np.array(val_indices_totali),
         test=np.array(test_indices_totali))

# Stampa controllo
logger.info("Train shape: %s %s", X_train.shape, Y_train.shape)
logger.info("Val shape: %s %s", X_val.shape, Y_val.shape)
logger.info("Test shape: %s %s", X_test.shape, Y_test.shape)


# Creo i dataset per il training val e test
train_dataset = HARDataset(X_train, Y_train)
val_dataset = HARDataset(X_val, Y_val)
test_dataset = HARDataset(X_test, Y_test)

 #stampo il dataset di training e di test a livello di dimensioni
logger.info("Lunghezza dataset di training: %s", len(train_dataset))
logger.info("Lunghezza dataset di validazione: %s", len(val_dataset))
logger.info("Lunghezza dataset di test: %s", len(test_dataset))


logger.debug(f"Tipo di train_dataset: {type(train_dataset)}")
logger.debug(f"Tipo di val_dataset: {type(val_dataset)}")
logger.debug(f"Tipo di test_dataset: {type(test_dataset)}")

BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model_inference_car_without_anything_timeseries.pkl")
BEST_SCORE_PATH = os.path.join(REPORTS_DIR, "best_score_inference_car_without_anything_timeseries.txt")

best_hyperparams_file = os.path.join(REPORTS_DIR, 'best_hyperparameters_inference_inference_car_without_anything_timeseries.csv')
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
        model.fc = nn.Linear(num_ftrs, 10)  # 4 classi
        model.set_n_classes(10)


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
    best_hyperparameters_df.to_csv(os.path.join(REPORTS_DIR, 'best_hyperparameters_inference_inference_car_without_anything_timeseries.csv'), index=False)

    best_lr = study.best_params['lr']
    best_batch_size = study.best_params['batch_size']

    #visualizzare la storia dell'ottimizzazione effettuata da Optuna. Ci permette di vedere come l'f1 score
    # è cambiato nel corso delle diverse prove (trials) durante l'ottimizzazione.
    file_name = "optimization_history_inference_car_without_anything_timeseries.png"
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
    Y=Y_train,
    batch_size=best_batch_size,
    figure_name="cm_train_best_hyp_inference_optuna_inference_car_without_anything_timeseries"
)

#STAMPO CM DELL'ALLENAMENTO SUL VAL SET
plot_CM(
    mdl_class=DeepConvLSTM,
    mdl_weights=BEST_MODEL_PATH,
    X=X_val,
    Y=Y_val,
    batch_size=best_batch_size,
    figure_name="cm_val_best_hyp_inference_optuna_inference_car_without_anything_timeseries"
)

# Creo modello e carico pesi del miglior modello (trovato prima in optuna)
model = DeepConvLSTM(n_classes=10)

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
test_loss, test_acc, test_f1 = train_with_cm.evaluate_model(model, test_loader, figure_name= "cm_test_best_hyp_optuna_inference_car_without_anything_timeseries", save_confusion_matrix=True)




