import numpy as np
from HumanActivityRecognition.app_config import RAW_DATA_DIR_TRAIN
from HumanActivityRecognition.app_config import RAW_DATA_DIR_TEST

from HumanActivityRecognition.utils import data_preprocessing

from HumanActivityRecognition.run_config import SLIDING_WINDOW_LENGTH
from HumanActivityRecognition.run_config import NB_SENSOR_CHANNELS
from HumanActivityRecognition.run_config import SLIDING_WINDOW_STEP

from HumanActivityRecognition import sliding_window_on_data

from models.DeepConvLSTM import DeepConvLSTM

from HumanActivityRecognition.utils import data_processing
from HumanActivityRecognition import init_weights
from HumanActivityRecognition import train
from HumanActivityRecognition.utils import data_analysis

#BUILD DATASET DI TRAIN E DI TEST 
datasetTracesTrain = data_preprocessing.build_dataset(RAW_DATA_DIR_TRAIN)
datasetTracesTest=data_preprocessing.build_dataset(RAW_DATA_DIR_TEST)

#stampe di verifica
print(f"Total number of traces in datasetTracesTrain: {len(datasetTracesTrain)}")
print(f"Total number of traces in datasetTracesTest: {len(datasetTracesTest)}")

first_trace=datasetTracesTrain[0]
print(f"Shape of first trace data: {first_trace['TraceData'].shape}")


#AGGIUNTA LABLE AI DATASET
dataset_train_labled=data_preprocessing.add_labels_to_dataset(datasetTracesTrain)
dataset_test_labled=data_preprocessing.add_labels_to_dataset(datasetTracesTest)


#DISTRIBUZIONE DATASET 
print(f"Distribuzione etichette dataset di train")
data_analysis.plot_label_distribution(dataset_train_labled)
print(f"Distribuzione etichette dataset di test")
data_analysis.plot_label_distribution(dataset_test_labled)




#stampe di verifica
first_trace_labled=dataset_train_labled[0]
print(f"TraceId: {first_trace_labled['TraceID']}")
print(f"Shape of first trace data LABLED: {first_trace_labled['TraceData'].shape}")


if dataset_train_labled:
    for i,row in enumerate(dataset_train_labled[0]['TraceData']):
        if i>=10:
            break
        print(",".join(map(str,row)))


#SLIDING WINDOW SUI DATI DI TRAIN E TEST, IN OUTPUT AVRO' UNA MATRICE X_TRAIN E X_TEST con le dimensioni attese per il modello

X_Train, Y_Train=sliding_window_on_data.apply_sliding_window(dataset_train_labled,SLIDING_WINDOW_LENGTH, SLIDING_WINDOW_STEP, NB_SENSOR_CHANNELS)

X_Test, Y_Test=sliding_window_on_data.apply_sliding_window(dataset_test_labled, SLIDING_WINDOW_LENGTH, SLIDING_WINDOW_STEP, NB_SENSOR_CHANNELS)

print(f"Distribuzione windows dataset di train")
data_analysis.plot_window_distribution(X_Train,Y_Train)
print(f"Distribuzione windows dataset di test")
data_analysis.plot_window_distribution(X_Test,Y_Test)


#Stampe di debug

print("X_train shape:", X_Train.shape)
print("Y_train shape:", Y_Train.shape)

print("X_Test shape:", X_Test.shape)
print("Y_Test shape:", Y_Test.shape)

#visualizza le prime righe di X_train e Y_train
print("\nPrime righe di X_train:")
print(X_Train[:5])  # Stampa le prime 5 righe di X_train

print("\nPrime righe di Y_train:")
print(Y_Train[:5])  # Stampa le prime 5 etichette di Y_train

# Stampa un riepilogo delle etichette uniche
unique_labels = np.unique(Y_Train)
print(f"\nEtichette uniche in Y_train: {unique_labels}")


#creo istanza modello

net= DeepConvLSTM()

#train

train.train(net, X_Train,Y_Train,X_Test,Y_Test,epochs=10,batch_size=84, lr=0.01)

