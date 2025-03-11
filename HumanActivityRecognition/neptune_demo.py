import neptune

#connetto il progetto a neptune e creo un run (esperimento)

run= neptune.init_run()

#Log hyperparameters ossia i parametri che voglio ottimizzare
run["parameters"] = {
    "batch_size': 64,"
    "learning_rate": 0.001,
    "epochs": 10
}

#Log dataset versions
