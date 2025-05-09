import numpy as np
from numpy.lib.stride_tricks import as_strided as ast
from utils.log_config import logger


def norm_shape(shape):
    '''
    Normalizza le forme degli array numpy, assicurandosi che siano sempre espresse come tuple.
    Anche se viene passato un numero intero, verrà restituita una tupla con un solo elemento.

    Parametri:
        shape - Può essere un intero oppure una tupla (o lista) di interi.

    Ritorna:
        Una tupla contenente la forma normalizzata.
    '''
    try:
        # Provo a convertire il valore in un intero
        i = int(shape)
        return (i,)  # Se è un intero, lo restituisco come una tupla di un solo elemento
    except TypeError:
        # Se il valore passato non è un numero, ignoro l'errore e passo al tentativo successivo
        pass
    
    try:
        # Provo a convertire il valore in una tupla
        t = tuple(shape)
        return t  # Se l'operazione ha successo, restituisco la tupla risultante
    except TypeError:
        # Se il valore passato non è una sequenza iterabile, ignoro l'errore e procedo
        pass
    
    # Se nessuno dei due tentativi ha avuto successo, sollevo un errore
    raise TypeError('shape deve essere un intero o una tupla di interi')





def sliding_window(a, ws, ss=None, flatten=True, min_pad_samples=30, extreme_pad_samples=10):
    '''
    Applica una finestra mobile (sliding window) su un array multidimensionale.
    Salva l'informazione relativa alla quantità di padding applicato per non perdere nessuna finestra.
    padding --> len(ws) - len(a) > 0.3*len(ws)
    padding estremo --> 0.1*len(ws) <= len(ws) - len(a) <= 0.3*len(ws)

    Se len(ws) - len(a) <= 0.1*len(ws) non viene applicato nessun padding e la finestra viene scartata.

    Parametri:
        a  - Un array numpy di n-dimensioni.
        ws - Un intero (se a è 1D) o una tupla (se a è 2D o superiore) che rappresenta la dimensione della finestra in ogni dimensione.
        ss - Un intero (se a è 1D) o una tupla (se a è 2D o superiore) che rappresenta l'entità dello spostamento della finestra in ogni dimensione.
             Se non specificato, assume il valore di ws.
        flatten - Se True, le finestre vengono appiattite in un array 1D, altrimenti viene mantenuta la forma multidimensionale.

    Ritorna:
        Un array contenente tutte le finestre n-dimensionali estratte da `a`.
        Un array contenente un codice relativo al padding applicato - {0: nessun padding, 1: padding, 2: padding estremo (applicato per non perdere aluna finestra per azioni molto piccole)}
    '''

    if ss is None:
        # Se ss non è fornito, la finestra non avrà sovrapposizioni.
        ss = ws
    
    # Normalizzo ws e ss per garantire che siano sempre tuple
    ws = norm_shape(ws)
    ss = norm_shape(ss)
    
    # Converto ws, ss e la forma dell'array `a` in array numpy per eseguire calcoli su tutte le dimensioni contemporaneamente.
    ws = np.array(ws)
    ss = np.array(ss)
    shape = np.array(a.shape)
    
    # Controllo che ws, ss e a.shape abbiano lo stesso numero di dimensioni: quindi tutti devono essere interi o array 2d ecc
    ls = [len(shape), len(ws), len(ss)]
    if 1 != len(set(ls)):
        raise ValueError(f'a.shape, ws e ss devono avere la stessa lunghezza. Valori ricevuti: {ls}')
    
    padding_code_vector = []

    # Se la lunghezza della finestra è maggiore del  numero di campioni disponibili, aggiungo padding
    if np.any(ws > shape):
        logger.debug(f"La lunghezza della finestra è maggiore della lunghezza dell'array. Applico padding.")
        num_actual_samples = len(a)  # Prendo i campioni disponibili
        if num_actual_samples < extreme_pad_samples:  
            # Scarto i campioni se sono inferiori al 30% della finestra
            logger.warning("Non ci sono abbastanza campioni per creare una finestra valida. Campioni scartati.")
        else:
            num_padding = ws[0] - num_actual_samples  # Quantità di padding necessaria
            padding_start = np.random.randint(1, num_padding) # Estraggo un numero casuale tra 1 e num_padding
            padding_end = num_padding - padding_start
            logger.info(f"Padding iniziale: {padding_start}, Padding finale: {padding_end}")
            # Aggiungo padding all'inizio e alla fine
            padded_samples = np.concatenate((np.zeros((padding_start,) + a.shape[1:]), a), axis=0) # Aggiungo padding all'inizio
            padded_samples = np.concatenate((padded_samples, np.zeros((padding_end,) + a.shape[1:])), axis=0) # Aggiungo padding alla fine
            a = padded_samples # Aggiorno l'array con i campioni aggiunti
            if num_actual_samples < min_pad_samples and num_actual_samples > extreme_pad_samples:
                padding_code_vector.append(2)
            elif num_actual_samples < ws[0]:
                padding_code_vector.append(1)
            logger.info(f"Nuova lunghezza array: {len(a)}")
            shape = np.array(a.shape) # Aggiorno la forma dell'array con i nuovi campioni aggiunti

    
    # Calcolo il numero di finestre che è possibile estrarre da `a` in ogni dimensione
    newshape = norm_shape(((shape - ws) // ss) + 1) #calcolo il numero di finestre in ogni dimensione che posso estrarre dall'array originale con passo ss e lunghezza ws, aggiungo 1 per considerare anche l'ultimo elemento, qui aveo quindi ad esempio (3finestre, 9 canali)
    
    #Aggiungo la dimensione della finestra alla forma calcolata prima, in modo che la fornma finale rappresenta sia il numero di finestre che le dimensioni effettive di ciascuna finestra => avro quindi ad esempio (3 finestre, 100 campioni ogni finestra, 9 canali)
    newshape += norm_shape(ws)
    
    #strides= distanza in byte tra gli elementi adiacenti lungo ogni dimensione. => stride mi dice dove gli trovano gli elementi successivi per ogni dimensione dell'array
    #se ad esempio a contineen float32 (4 byte), ogni riga ha 100 colonne e 9 canali => disntaza tra due righe consetuive è 100*9*4=3600 byte (righe), dimensione 1 = ogni colonna ha 9 canali quindi la distamza tra due colonne consecutive nella stessa riga è 9*4=36 byte, dimensione 2 (canali) 4 byte (distanza tra due canali consecutivi nella stessa posizione di riga e colonna)
    #quindi strides originale è (3600, 36, 4)
    #ss= passo della finestra , poi sommo il passo della finestra per ogni dimensione in modo da ottenere la distanza tra due finestre consecutive
    #quindi se ad esempio ho 100 campioni e passo della finestra è 50, la distanza tra due finestre consecutive sarà 50*9*4=1800 byte (righe), 50*4=200 byte (colonne), 50*4=200 byte (canali)
    newstrides = norm_shape(np.array(a.strides) * ss) + a.strides
    
    # Creo l'array strided con la nuova forma e i nuovi stride questo contiene tutte le finestre estratte dall'array originale (con una vista sugli stessi dati non una copia) 
    strided = ast(a, shape=newshape, strides=newstrides)
    #stampo il numero di finestre estratte
    logger.info(f"Numero di finestre estratte: {strided.shape[0]}")

    # Verifico se rimangono campioni dopo l'ultima finestra completa
    '''differenza tra numero totale di campioni (shape[0]) e la quantità di campioni che sono stati gia "coperti" dalle finestre mobili complete.
    -newshape[0]-1 *ss[0] : calcola la distanza in termini di campioni che c'è tra la prima finestra e l'ultima finestra estratta, tenendo conto del passo ss[0]. 
    -newshape[0] - 1 è il numero di "passi" tra la prima e l'ultima finestra. Ogni passo è lungo ss[0], che rappresenta la distanza in campioni tra due finestre consecutive.
    - ws[0]: Questo sottrae la lunghezza della finestra (ws[0]) perché, una volta che l'ultima finestra è stata estratta, la sua lunghezza non è più necessaria nel calcolo dei campioni rimanenti.'''
    remaining_samples = (shape[0] - ((newshape[0] - 1) * ss[0]) - ws[0]) 
    logger.debug(f"Campioni rimanenti dopo l'ultima finestra completa: {remaining_samples}")
    if remaining_samples > 0:
        if remaining_samples > extreme_pad_samples:
            num_padding = ws[0] - remaining_samples 
            padding_start = np.random.randint(1, num_padding)
            padding_end = num_padding - padding_start
            last_window = np.concatenate((a[-remaining_samples:], np.zeros((padding_end, a.shape[1]))), axis=0)
            last_window = np.concatenate((np.zeros((padding_start, a.shape[1])), last_window), axis=0)
            last_window = last_window[None, :, :]

            strided = np.concatenate((strided, last_window[None, :]), axis=0)
            newshape = strided.shape

            if remaining_samples < min_pad_samples and remaining_samples > extreme_pad_samples:
                padding_code_vector.append(2)
                logger.info(f"Padding estremo applicato. Padding code: 2")
            elif remaining_samples < ws[0]:
                padding_code_vector.append(1)
                logger.info(f"Padding normale applicato. Padding code: 1")
            
            logger.debug(f"Nuova finestra con padding: {last_window.shape}")
        else:
            logger.info("Non è stato applicato padding perché i campioni rimanenti sono troppo pochi.")
    
    if not flatten:
        return strided
    
    # Se flatten è True, riduco le dimensioni dell'array trasformandolo in una lista piatta di finestre anizhcè una lista di finestre multidimensionali
    meat = len(ws) if ws.shape else 0 #numero di dimensioni della finestra
    firstdim = (np.prod(newshape[:-meat]),) if ws.shape else () #prodotto delle dimensioni della finestra (esclusa la dimensione delle finestre) => mi restituisce il numero di finestre
    dim = firstdim + tuple(newshape[-meat:]) #aggiungo le dimensioni della finestra => mi restituisce il numero di finestre e le dimensioni della finestra
    strided = strided.reshape(dim) #riduco le dimensioni dell'array trasformandolo in una lista piatta di finestre
    logger.info(f"Numero totale di finestre (dopo padding finale): {strided.shape[0]}")
    return strided, padding_code_vector




 
def old_sliding_window(a,ws,ss = None,flatten = True):
    '''
    Return a sliding window over a in any number of dimensions

    Parameters:
        a  - an n-dimensional numpy array
        ws - an int (a is 1D) or tuple (a is 2D or greater) representing the size
             of each dimension of the window
        ss - an int (a is 1D) or tuple (a is 2D or greater) representing the
             amount to slide the window in each dimension. If not specified, it
             defaults to ws.
        flatten - if True, all slices are flattened, otherwise, there is an
                  extra dimension for each dimension of the input.

    Returns
        an array containing each n-dimensional window from a
    '''

    if None is ss:
        # ss was not provided. the windows will not overlap in any direction.
        ss = ws
    ws = norm_shape(ws)
    ss = norm_shape(ss)

    # convert ws, ss, and a.shape to numpy arrays so that we can do math in every
    # dimension at once.
    ws = np.array(ws)
    ss = np.array(ss)
    shape = np.array(a.shape)


    # ensure that ws, ss, and a.shape all have the same number of dimensions
    ls = [len(shape),len(ws),len(ss)]
    if 1 != len(set(ls)):
        raise ValueError(\
        'a.shape, ws and ss must all have the same length. They were %s' % str(ls))

    # ensure that ws is smaller than a in every dimension
    if np.any(ws > shape):
        raise ValueError(\
        'ws cannot be larger than a in any dimension.\
 a.shape was %s and ws was %s' % (str(a.shape),str(ws)))

    # how many slices will there be in each dimension?
    newshape = norm_shape(((shape - ws) // ss) + 1)
    # the shape of the strided array will be the number of slices in each dimension
    # plus the shape of the window (tuple addition)
    newshape += norm_shape(ws)
    # the strides tuple will be the array's strides multiplied by step size, plus
    # the array's strides (tuple addition)
    newstrides = norm_shape(np.array(a.strides) * ss) + a.strides
    strided = ast(a,shape = newshape,strides = newstrides)
    if not flatten:
        return strided

    # Collapse strided so that it has one more dimension than the window.  I.e.,
    # the new array is a flat list of slices.
    meat = len(ws) if ws.shape else 0
    firstdim = (np.prod(newshape[:-meat]),) if ws.shape else ()
    dim = firstdim + (newshape[-meat:])
    # remove any dimensions with size 1
#     dim = filter(lambda i : i != 1,dim)
    return strided.reshape(dim)
