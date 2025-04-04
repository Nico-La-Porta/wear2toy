import os
import sys
import numpy as np
import torch
from app_config import PROJ_ROOT, RAW_DATA_DIR_TRAIN, RAW_DATA_DIR_TEST, REPORTS_DIR, FIGURES_DIR, MODELS_DIR
import pandas as pd
import matplotlib.pyplot as plt
import data_analysis
sys.path.append(os.path.join(PROJ_ROOT, "HumanActivityRecognition"))
from utils import data_preprocessing, init_weights
from utils.log_config import *
from run_config import SLIDING_WINDOW_LENGTH, NB_SENSOR_CHANNELS, SLIDING_WINDOW_STEP
import sliding_window_on_data
from torch.utils.data import DataLoader
from models.DeepConvLSTM import DeepConvLSTM, HARDataset
from sklearn.preprocessing import StandardScaler
import train
import train_with_cm
import optuna
import optuna.visualization as vis
from optuna.pruners import MedianPruner


# Impostazione il seed per la riproducibilità
init_weights.set_seed(42)
    
# Prepara i dati
datasetTracesTrain = data_preprocessing.build_dataset(RAW_DATA_DIR_TRAIN)
datasetTracesTest = data_preprocessing.build_dataset(RAW_DATA_DIR_TEST)


dataset_train_labled = data_preprocessing.add_labels_to_dataset(datasetTracesTrain)
dataset_test_labled = data_preprocessing.add_labels_to_dataset(datasetTracesTest)


classees_to_remove= [15,16,19,21]

dataset_train_labled= data_preprocessing.remove_classes(dataset_train_labled, classees_to_remove)
dataset_test_labled= data_preprocessing.remove_classes(dataset_test_labled, classees_to_remove)




grouped_activity_data = data_preprocessing.combine_and_group(dataset_train_labled, dataset_test_labled)





# Calcolo KS test e statistiche per ogni attività
#data_preprocessing.check_normality_and_save_by_activity(grouped_activity_data, FIGURES_DIR)


#carico i risultati del test di normalità 
json_results = data_preprocessing.load_json_results(FIGURES_DIR)

#applico gli scalers ai dati di train e di test
scaled_train_data = data_preprocessing.apply_scalers_to_dataset(dataset_train_labled, json_results, FIGURES_DIR)

logger.info("Train data scalers applied")
#verifico media e deviazione standard


scaled_test_data = data_preprocessing.apply_scalers_to_dataset(dataset_test_labled, json_results, FIGURES_DIR)
logger.info("Test data scalers applied")


logger.debug("Verifico se lo scaler è stato applicato correttamente")
data_preprocessing.check_scaled_data(dataset_train_labled, scaled_train_data, "Train")
data_preprocessing.check_scaled_data(dataset_test_labled, scaled_test_data, "Test")


X_Train, Y_Train = sliding_window_on_data.apply_sliding_window(scaled_train_data, SLIDING_WINDOW_LENGTH, SLIDING_WINDOW_STEP, NB_SENSOR_CHANNELS)
X_Test, Y_Test = sliding_window_on_data.apply_sliding_window(scaled_test_data, SLIDING_WINDOW_LENGTH, SLIDING_WINDOW_STEP, NB_SENSOR_CHANNELS)

#verifico che le classi siano state rimosse correttamente
#stampo i valori unici di y_train e y_test
logger.debug("Classi in Y_train: %s", np.unique(Y_Train))
logger.debug("Classi in Y_test: %s", np.unique(Y_Test))





# Creazione dataset
train_dataset = HARDataset(X_Train, Y_Train)
test_dataset = HARDataset(X_Test, Y_Test)

best_hyperparams_file=os.path.join(REPORTS_DIR, 'best_hyperparameters_dl_norm_without_sampler_removed_4_classes.csv')
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
    best_hyperparameters_df.to_csv(os.path.join(REPORTS_DIR, 'best_hyperparameters_dl_norm_without_sampler_removed_4_classes.csv'), index=False)

    print("Migliori iperparametri trovati e salvati:", best_hyperparameters)
    best_lr = study.best_params['lr']
    best_batch_size = study.best_params['batch_size']
    #visualizzare la storia dell'ottimizzazione effettuata da Optuna. Ci permette di vedere come l'f1 score
    # è cambiato nel corso delle diverse prove (trials) durante l'ottimizzazione.
    file_name = "optimization_history_dl_norm_without_sampler_removed_4_classes.png"
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
best_f1_score = train_with_cm.train(best_net, train_loader, test_loader, epochs=100, batch_size=best_batch_size, lr=best_lr, figure_name="model_dl_norm_without_sampler_removed_4_classes.png")

# Salva il miglior modello
model_save_path = os.path.join(MODELS_DIR, 'best_model_dl_norm_without_sampler_removed_4_classes.pth')
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)
torch.save(best_net.state_dict(), model_save_path)

print(f"Best model trained with the optimal hyperparameters and saved at {model_save_path}")