import numpy as np

def sift_svm(image, detector, tipo):
    '''
    image -> imagen a procesar
    detector -> objeto SIFT ya creado con nfeatures=X
    tipo -> indica el tipo de vector de características a extraer
    
       Tipos disponibles:
        1. Estadísticas globales (media, std, max)
    
    '''

    _, descriptors = detector.detectAndCompute(image, None)
    # Tipo 1 – Estadísticas: media, std
    if tipo == 1:
        if descriptors is None or descriptors.shape[0] == 0:
            return np.zeros(128 * 3)

        mean_desc = np.mean(descriptors, axis=0)
        std_desc = np.std(descriptors, axis=0)
        max_desc = np.max(descriptors, axis=0)

        return np.concatenate([mean_desc, std_desc, max_desc])


    else:
        raise ValueError(f"Tipo de extracción no válido: {tipo}")
