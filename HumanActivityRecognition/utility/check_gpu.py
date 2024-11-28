import torch

#verifico se la GPU è disponibile
def check_gpu_availability():

    if torch.cuda.is_available():
        print('Training on GPU!')
        return True
    else:
        print('No GPU available, training on CPU; consider making n_epochs very small.')
        return False

