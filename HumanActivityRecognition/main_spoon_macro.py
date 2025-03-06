import os
import pandas as pd
import numpy as np
from app_config import REPORTS_DIR, FIGURES_DIR, MODELS_DIR
import sliding_window_on_data
from torch.utils.data import DataLoader

from models.DeepConvLSTM import DeepConvLSTM, HARDataset 
import optuna
import optuna.visualization as vis
import train
import torch
import train_with_cm
import matplotlib.pyplot as plt
#definisco il path da cui leggere i .csv

path='C:\codes\HumanActivityRecognition\data\pdd_data'
print(path)




#Visto che l'informazione relativa all'id del bambino lo abbiamo, così come abbiamo anche l'informazione relativa
#al giocattolo, vado a filtrare i .csv in modo tale da avere solo le righe che hanno l'attività non nulla e poi
#appendo tutte le righe delle righe non nulle in un unico dataframe
#per tutti i file che terminano in .csv nella cartella path
# Definisco il percorso della cartella contenente i CSV


# Nome del file CSV finale
final_csv_path = os.path.join(path, 'df_SP_non_null.csv')

# Controllo se il file esiste già
if os.path.exists(final_csv_path):
    print(f"Il file {final_csv_path} esiste già. Lo sto caricando...")
    df_spoon = pd.read_csv(final_csv_path)
else:

    df_list_spoon = [] #lista vuota per appendere i dataframe con attività non nulla
    for file in os.listdir(path):
        if file.endswith('.csv'):
            #leggo solo i file che dopo l'undescore ha BA
            if file.split('_')[1]=='SP.csv':
                df_temp=pd.read_csv(os.path.join(path,file))
                df_temp = df_temp[df_temp['action_id'] != 0]
                df_list_spoon.append(df_temp)

    df_spoon = pd.concat(df_list_spoon)
    print("Dimensioni del df_spoon con tutte le attività non nulle")
    print(df_spoon.shape)
    print(df_spoon.columns)
    print(df_spoon['action'].value_counts())

    #salvo il dataframe
    df_spoon.to_csv(final_csv_path,index=False)
    print(f"Salvato il dataframe df_SP_non_null.csv")


#ora divido il dataframe in base all'attività (action_id) e salvo i dataframe in un file .csv
#per ogni attività

for action_id in df_spoon['action_id'].unique():
    df_action = df_spoon[df_spoon["action_id"] == action_id] #filtro il dataframe in base all'attività
    print(f"Dimensioni del dataframe df_spoon_action_{action_id}")
    print(df_action.shape) #stampo le dimensioni del dataframe
    print(df_action.columns) #stampo le colonne del dataframe
    print(df_action['action'].value_counts()) #stampo il conteggio delle attività
    #salvo il dataframe
    df_action.to_csv(os.path.join(path,f'df_spoon_action_{action_id}.csv'),index=False) #index=False per non salvare l'indice
    print(f"Salvato il dataframe df_spoon_action_{action_id}.csv")


#applico sliding window con la funzion process_csv
#definisco i parametri
nb_sensor_channels = 13
sliding_window_length = 30
sliding_window_step = 20

#ora applico la funzione sliding window (che mi da come output x_window e y_window) a tutti i .csv relativi al toy palla
#e poi concateno tutto in un unica x_train, y_train

X= []
Y= []

for file in os.listdir(path):
    if file.endswith('.csv') and file.split('_')[1] == 'spoon':
        file_path = os.path.join(path, file)
        print(file_path)
        X_windows, Y_windows = sliding_window_on_data.process_csv(file_path, nb_sensor_channels, sliding_window_length, sliding_window_step)
        X.append(X_windows)
        Y.append(Y_windows)

# Concatenate all the windows into a single array
X = np.concatenate(X, axis=0)
Y = np.concatenate(Y, axis=0)

#NUMERO TOTALE DI FINESTRE PER LA PALLA
print("Numero totale di finestre per il giocattolo spoon:")
print(X.shape)
print(Y.shape)

#verifca
print(X)
print(Y)


# estraggo gli id dei bambini per vedere quanti ne ho
kid_ids = X[:, :, -2]  # Assuming kid_id is the third last column

# Get unique kid_ids
unique_kid_ids = np.unique(kid_ids)
print("Unique kid_ids in X:")
print(unique_kid_ids)



#Splitto il dataset in base al numero di azioni eseguite per avere congruenza temporale tra train e test e per 
#cercare di bilanciare le finestre in train e test

#filtro per ogni
#filtro per contare il numero di finestre per ogni azione
unique_actions, counts = np.unique(Y, return_counts=True)
print(unique_actions)
action_counts = dict(zip(unique_actions, counts))

print("Numero di finestre per ogni azione:")
for action, count in action_counts.items():
    print(f"Azione {action}: {count} finestre")


#split ratio  (70% nel train e 30% nel test)
split_ratio = 0.7

# Split the data
X_train = []
Y_train = []
X_test = []
Y_test = []

for action in unique_actions:

    #trovo gli indici delle finestre corrispondenti a ciascuna azione
    action_indices = np.where(Y == action)[0]
    print(action_indices)

    # Calcolo il numero di finestre da usare per il train e per il test
    num_windows = len(action_indices)
    num_train = int(num_windows * split_ratio)
    num_test = num_windows - num_train
    
    # Divido gli indici delle finestre in train e test
    train_indices = action_indices[:num_train]
    test_indices = action_indices[-num_test:]
    
    # Aggiungo le finestre al train e al test set
    X_train.append(X[train_indices])
    Y_train.append(Y[train_indices])
    X_test.append(X[test_indices])
    Y_test.append(Y[test_indices])

# Concateno tutti i dati in un unico array
X_train = np.concatenate(X_train, axis=0)
Y_train = np.concatenate(Y_train, axis=0)
X_test = np.concatenate(X_test, axis=0)
Y_test = np.concatenate(Y_test, axis=0)

print("Training set shape:", X_train.shape, Y_train.shape)
print (Y_train)
print("Test set shape:", X_test.shape, Y_test.shape)
print(Y_test)

#stampo il tipo di valore che contiene x_train e y_train(se int float ecc)
print("Tipo di X_train:", X_train.dtype)
print("Tipo di Y_train:", Y_train.dtype)

#stampo il tipo di x_train y train x test e y test
print("Tipo di X_train:", type(X_train))
print("Tipo di Y_train:", type(Y_train))
print("Tipo di X_test:", type(X_test))
print("Tipo di Y_test:", type(Y_test))

Y_train = Y_train.flatten()
Y_test = Y_test.flatten()

print("Etichette train:", Y_train)
print("Etichette test:", Y_test)

#tipo
print("Tipo di Y_train:", type(Y_train))
print("Tipo di Y_test:", type(Y_test))

# Trova tutte le etichette uniche presenti nei dati
unique_labels = np.unique(Y_train)

# Crea un dizionario che mappa ogni etichetta originale a un valore consecutivo
label_mapping = {label: idx for idx, label in enumerate(unique_labels)}

# Stampa il dizionario per vedere il mapping
print("Mapping delle etichette:", label_mapping)

# Applica il mapping ai dataset di train e test
Y_train_mapped = np.array([label_mapping[y] for y in Y_train])
Y_test_mapped = np.array([label_mapping[y] for y in Y_test])

# Controllo finale
print("Nuove etichette train:", np.unique(Y_train_mapped))
print("Nuove etichette test:", np.unique(Y_test_mapped))


# Creo i dataset per il training e il test
train_dataset = HARDataset(X_train, Y_train_mapped)
test_dataset = HARDataset(X_test, Y_test_mapped)

 #stampo il dataset di training e di test a livello di dimensioni
print("Dataset di training:", len(train_dataset))
print("Dataset di test:", len(test_dataset))

print("Tipo di train_dataset:", type(train_dataset))
print("Tipo di test_dataset:", type(test_dataset))






def objective(trial):
    # Definisci gli iperparametri da ottimizzare
    lr = trial.suggest_float('lr', 1e-4, 1e-1, log=True)
    batch_size = trial.suggest_categorical('batch_size', [4, 8, 12, 16])

    # Crea i DataLoader con il batch_size suggerito
    #runno di nuovo il train con 5 secondi di finestra 
    train_loader = DataLoader(train_dataset, batch_size=batch_size,drop_last=True, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    # Crea il modello con gli iperparametri suggeriti
    net = DeepConvLSTM(n_classes=len(unique_labels), nb_sensor_channels=13, sliding_window_length=30)

    # Esegui l'allenamento
    best_f1_score = train.train(net, train_loader, test_loader, epochs=100, batch_size=batch_size, lr=lr, patience=7, f1_average='macro')
    
    return best_f1_score

# Creazione studio Optuna ottimizza, nel senso di minimizzare la loss in 100 prove
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

print("Best hyperparameters: ", study.best_params)
print("Highest F1-score: ", study.best_value)


#salvo i best hyperparameters
best_hyperparameters = study.best_params
best_hyperparameters['best_f1_score'] = study.best_value
best_hyperparameters_df = pd.DataFrame([best_hyperparameters])
best_hyperparameters_df.to_csv(os.path.join(REPORTS_DIR, 'best_hyperparameters_spoon_without_sampler_macro.csv'), index=False)





#visualizzare la storia dell'ottimizzazione effettuata da Optuna. Ci permette di vedere come l'f1 score
# è cambiato nel corso delle diverse prove (trials) durante l'ottimizzazione.
file_name = "optimization_history_spoon_without_sampler_macro.png"
fig=vis.plot_optimization_history(study)
plt.show()

# Salvo il grafico nella cartella FIGURES con il nome specificato
fig.write_image(os.path.join(FIGURES_DIR, file_name))

print(f"Grafico salvato in figures /{file_name}")


# Ora crea e allena il modello con i migliori iperparametri
best_lr = study.best_params['lr']
best_batch_size = study.best_params['batch_size']

# Crea i DataLoader con i migliori iperparametri
train_loader = DataLoader(train_dataset, batch_size=best_batch_size, drop_last=True, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=best_batch_size, shuffle=False, drop_last=True)

# Crea il modello con i migliori iperparametri
best_net = DeepConvLSTM(n_classes=len(unique_labels), nb_sensor_channels=13, sliding_window_length=30)

# Esegui l'allenamento con i migliori iperparametri
best_f1_score = train_with_cm.train(best_net, train_loader, test_loader, epochs=100, batch_size=best_batch_size, lr=best_lr, figure_name="model_spoon_without_sampler_macro", patience=7, f1_average='macro')

# Salva il miglior modello
model_save_path = os.path.join(MODELS_DIR, 'best_model_spoon_without_sampler_macro.pth')
torch.save(best_net.state_dict(), model_save_path)

print(f"Best model trained with the optimal hyperparameters and saved at {model_save_path}")