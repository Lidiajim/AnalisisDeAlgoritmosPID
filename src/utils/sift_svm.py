import numpy as np
import algoritmos.sift as sf

def sift_svm(image, parametros):
    '''
    image -> imagen a procesar
    parametros -> diccionario con los parámetros del detector SIFT

    ########################################## PASOS DEL ALGORITMO ##########################################
        1. Inicializamos el detector SIFT con los parámetros dados.
        2. Detectamos los puntos clave y descriptores en la imagen.
        3. Extraemos estadísticas de los descriptores.
    
    El objetivo de este código es facilitarle al SVM un método para la extracción
    de características aplicando el algoritmo SIFT.
    '''
    # Inicializamos el detector SIFT
    sift_detector = sf.sift_detect(
        nfeatures=parametros.get("nfeatures", 0),
        nOctaveLayers=parametros.get("nOctaveLayers", 3),
        contrastThreshold=parametros.get("contrastThreshold", 0.04),
        edgeThreshold=parametros.get("edgeThreshold", 10),
        sigma=parametros.get("sigma", 1.6)
    )
    
    # Detectamos los puntos clave y los descriptores
    keypoints, descriptors = sift_detector.sift.detectAndCompute(image, None)
    
    if descriptors is None or len(descriptors) == 0:
        return [0, 0, 0]  # Si no hay descriptores, devolvemos valores neutros
    
    # Calculamos estadísticas de los descriptores
    mean = np.mean(descriptors)
    std = np.std(descriptors)
    max_val = np.max(descriptors)
    
    return [mean, std, max_val]

