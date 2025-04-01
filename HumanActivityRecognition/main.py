import os
import sys
import numpy as np
import torch
from app_config import PROJ_ROOT, RAW_DATA_DIR_TRAIN, RAW_DATA_DIR_TEST, REPORTS_DIR, FIGURES_DIR, MODELS_DIR
import pandas as pd
import data_analysis
sys.path.append(os.path.join(PROJ_ROOT, "HumanActivityRecognition"))


from utils import data_preprocessing
from utils.log_config import logger

from run_config import SLIDING_WINDOW_LENGTH
from run_config import NB_SENSOR_CHANNELS
from run_config import SLIDING_WINDOW_STEP

import sliding_window_on_data
from torch.utils.data import DataLoader
from models.DeepConvLSTM import DeepConvLSTM, HARDataset
from sklearn.preprocessing import StandardScaler

from utils import init_weights
import train
import train_with_cm

import optuna
import optuna.visualization as vis
from optuna.pruners import MedianPruner
import matplotlib.pyplot as plt

# Impostazione il seed per la riproducibilità
init_weights.set_seed(42)
    
# Prepara i dati
datasetTracesTrain = data_preprocessing.build_dataset(RAW_DATA_DIR_TRAIN)
datasetTracesTest = data_preprocessing.build_dataset(RAW_DATA_DIR_TEST)
dataset_train_labled = data_preprocessing.add_labels_to_dataset(datasetTracesTrain)
dataset_test_labled = data_preprocessing.add_labels_to_dataset(datasetTracesTest)

grouped_activity_data = data_preprocessing.combine_and_group(dataset_train_labled, dataset_test_labled)

# Visualizzo le attività e il numero di tracce per attività
for activity, traces in grouped_activity_data.items():
    logger.info(f"Attività {activity} con {len(traces)} tracce")
    combined_data = np.concatenate(traces, axis=0)  # Unisco tutte le tracce per l'attività
    # Applico il test KS e traccio l'istogramma per ciascuna caratteristica
    for feature_idx in range(combined_data.shape[1]):
        feature_data = combined_data[:, feature_idx]  # Dati per una singola caratteristica

    #calcolo istogramma
        #plt.figure(figsize=(8, 6))
        #plt.hist(feature_data.flatten(), bins='auto', color='blue', alpha=0.7, label=f'Activity {activity}')
        #plt.title(f'Istogramma dei dati per l\'attività {activity} e la caratteristica {feature_idx}')
        #plt.xlabel('Valore dei dati')
        #plt.ylabel('Frequenza')
        #plt.legend(loc='best')

        #creo una sottocartella in FIGURES
        #if not os.path.exists(os.path.join(FIGURES_DIR, f"activity_{activity}")): # Se non esiste la cartella, la creo
            #os.makedirs(os.path.join(FIGURES_DIR, f"activity_{activity}")) # Creo la cartella
        #salvo l'istogramma
        #plt.savefig(os.path.join(FIGURES_DIR, f"activity_{activity}", f"histogram_feature_{feature_idx}.png")) # Salvo l'istogramma IN FIGURES/activity_{activity}
        #chiudo il plot
        #plt.close()

        # Eseguo il KS test
        stat, p_value = data_analysis.kstest(feature_data.flatten(), 'norm')  # KS test
        logger.info(f'Attività {activity} - p-value per il KS test per la caratteristica {feature_idx}: {p_value}')

        if p_value < 0.05:
            logger.info(f"I dati per l'attività {activity} e la caratteristica {feature_idx} non seguono una distribuzione normale (rifiutata dal KS test).")

        else:
            logger.info(f"I dati per l'attività {activity} e la caratteristica {feature_idx} seguono una distribuzione normale (non rifiutata dal KS test).")








#Calcolo e salvo le statistiche descrittive
stats = data_analysis.compute_descriptive_statistics(grouped_activity_data)


X_Train, Y_Train = sliding_window_on_data.apply_sliding_window(dataset_train_labled, SLIDING_WINDOW_LENGTH, SLIDING_WINDOW_STEP, NB_SENSOR_CHANNELS)
X_Test, Y_Test = sliding_window_on_data.apply_sliding_window(dataset_test_labled, SLIDING_WINDOW_LENGTH, SLIDING_WINDOW_STEP, NB_SENSOR_CHANNELS)


# Creo l'oggetto StandardScaler
scaler = StandardScaler()

# Reshape dei dati per normalizzare su tutte le dimensioni insieme
X_Train_reshaped = X_Train.reshape(-1, X_Train.shape[2])  # (numfinestre * ws, channels)
X_Test_reshaped = X_Test.reshape(-1, X_Test.shape[2])  # (numfinestre * ws, channels)

# Normalizzo i dati di train e test
X_Train_scaled_reshaped = scaler.fit_transform(X_Train_reshaped)  # Normalizzo i dati di train
X_Test_scaled_reshaped = scaler.transform(X_Test_reshaped)  # Normalizzo i dati di test

# Reshape indietro per tornare alla forma originale
X_Train_scaled = X_Train_scaled_reshaped.reshape(X_Train.shape)  # (numfinestre, ws, channels)
X_Test_scaled = X_Test_scaled_reshaped.reshape(X_Test.shape)  # (numfinestre, ws, channels)

logger.debug(f"Dimensioni X_Train_scaled: {X_Train_scaled.shape}")
logger.debug(f"Dimensioni X_Test_scaled: {X_Test_scaled.shape}")



logger.info("Dati normalizzati")
logger.debug(f"Media: {scaler.mean_}")
logger.debug(f"Deviazione standard: {scaler.scale_}")



# Creazione dataset
train_dataset = HARDataset(X_Train, Y_Train)
test_dataset = HARDataset(X_Test, Y_Test)

best_hyperparams_file=os.path.join(REPORTS_DIR, 'best_hyperparameters_dl_norm_without_sampler.csv')
if os.path.exists(best_hyperparams_file):
    print("Carico i migliori iperparametri da ", best_hyperparams_file)
    best_hyperparameters=pd.read_csv(best_hyperparams_file).iloc[0].to_dict()
    best_lr = best_hyperparameters['lr']
    best_batch_size = int(best_hyperparameters['batch_size'])

else:
    print("Nessun file di iperparametri trovato, eseguo l'ottimizzazione")

    def objective(trial):
        # Definisci gli iperparametri da ottimizzare
        lr = trial.suggest_float('lr', 1e-4, 1e-1, log=True)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])

        # Crea i DataLoader con il batch_size suggerito
        #runno di nuovo il train con 5 secondi di finestra 
        train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)


        # Crea il modello con gli iperparametri suggeriti
        net = DeepConvLSTM()

        # Esegui l'allenamento
        best_f1_score = train.train(net, train_loader, test_loader, epochs=100, batch_size=batch_size, lr=lr)
        
        return best_f1_score

    # Creazione studio Optuna 
    # Creazione dello studio con il MedianPruner
    study = optuna.create_study(
        direction='maximize', 
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10)  # Parametri di pruning per evitare di continuare trial non promettenti: n_startup_trials=5 significa che i primi 5 trial non verranno prunati, n_warmup_steps=10 significa che dopo 10 trial verrà applicato il pruning
        #Il pruner interromperà automaticamente i trial che non sono promettenti, basandosi sui punteggi parziali (F1-score) ottenuti durante l'allenamento.
    )
    study.optimize(objective, n_trials=100)

    print("Best hyperparameters: ", study.best_params)
    print("Highest F1-score: ", study.best_value)

    #salvo i best hyperparameters
    best_hyperparameters = study.best_params
    best_hyperparameters['best_f1_score'] = study.best_value
    best_hyperparameters_df = pd.DataFrame([best_hyperparameters])
    best_hyperparameters_df.to_csv(os.path.join(REPORTS_DIR, 'best_hyperparameters_dl_without_sampler.csv'), index=False)

    #visualizzare la storia dell'ottimizzazione effettuata da Optuna. Ci permette di vedere come l'f1 score
    # è cambiato nel corso delle diverse prove (trials) durante l'ottimizzazione.
    file_name = "optimization_history_dl_without_sampler.png"
    fig=vis.plot_optimization_history(study)
    plt.show()

    # Salvo il grafico nella cartella FIGURES con il nome specificato
    fig.write_image(os.path.join(FIGURES_DIR, file_name))

    print(f"Grafico salvato in figures /{file_name}")




# Crea i DataLoader con i migliori iperparametri
train_loader = DataLoader(train_dataset, batch_size=best_batch_size, drop_last=True, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=best_batch_size, shuffle=False, drop_last=True)

# Crea il modello con i migliori iperparametri
best_net = DeepConvLSTM()

# Esegui l'allenamento con i migliori iperparametri
best_f1_score = train_with_cm.train(best_net, train_loader, test_loader, epochs=100, batch_size=best_batch_size, lr=best_lr, figure_name="model_dl_norm_without_sampler")

# Salva il miglior modello
model_save_path = os.path.join(MODELS_DIR, 'best_model_dl_norm_without_sampler.pth')
torch.save(best_net.state_dict(), model_save_path)

print(f"Best model trained with the optimal hyperparameters and saved at {model_save_path}")