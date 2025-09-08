import neptune
import numpy as np
from time import sleep

run= neptune.init_run(
    project="carolina/first-example", #nome del progetto
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIwOGNhNjk1Zi02ZDI1LTQ4ZTMtOWJlMi0yOTE1ZDA3NGMzNzcifQ==", #token di autenticazione
    capture_hardware_metrics=True, #permette di catturare le metriche hardware
    capture_stderr=True, #permette di catturare le metriche di errore
    capture_stdout=True #permette di catturare le metriche di output
)


#log score
run["score"] = 0.62 #valore di esempio

#con questo for simulo il tempo di esecuzione e loggo delle metriche random
for i in range(100):
    sleep(0.2) #simula il tempo di esecuzione
    run["random_training_metric"].append(i*np.random.random()) #log delle metriche random
    run["other_random_training_metric"].append(0.5*i*np.random.random()) #log delle metriche random

run.stop()