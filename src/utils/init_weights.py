import torch
import torch.nn as nn
import numpy as np
import random
def set_seed(seed):
    torch.manual_seed(seed) #per tutte le operazioni che utilizzano il generatore di numeri casuali di PyTorch
    torch.cuda.manual_seed_all(seed) #se si usa gpu
    np.random.seed(seed) #per la libreria numpy
    random.seed(seed) #per la libreria di Python random
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_weights(m):
    if type(m) == nn.LSTM:
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
    elif type(m) == nn.Conv1d or type(m) == nn.Linear:
        torch.nn.init.orthogonal_(m.weight)
        m.bias.data.fill_(0)
  