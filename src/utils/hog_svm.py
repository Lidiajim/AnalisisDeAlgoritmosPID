import numpy as np
import algoritmos.hog as hog

def hog_svm(image, parametros):
    """
    image -> Imagen a procesar (en color o en escala de grises)
    parametros -> Diccionario con los parámetros del detector HOG

    ########################################## PASOS DEL ALGORITMO ##########################################
        1. Inicializamos el detector HOG con los parámetros dados.
        2. Calculamos el descriptor HOG para la imagen.
        3. Extraemos estadísticas del descriptor (media, desviación estándar y máximo).

    El objetivo de este código es proporcionar al SVM un método para la extracción
    de características aplicando el algoritmo HOG.
    """
    # Inicializamos el detector HOG utilizando los parámetros proporcionados o valores por defecto.
    hog_detector = hog.HOGDetect(
        cell_size      = parametros.get("cell_size", (8, 8)),
        block_size     = parametros.get("block_size", (2, 2)),
        block_stride   = parametros.get("block_stride", (4, 4)),
        nbins          = parametros.get("nbins", 9),
        win_stride     = parametros.get("win_stride", (8, 8)),
        padding        = parametros.get("padding", (8, 8)),
        scale          = parametros.get("scale", 1.05),
        bin_width      = parametros.get("bin_width", 20)
    )
    
    # Calcular el descriptor HOG para la imagen.
    # La función compute_hog se encarga de convertir la imagen a escala de grises y de calcular los gradientes,
    # histogramas por celda, y normalización de bloques para formar el descriptor.
    descriptor = hog_detector.compute_hog(image)
    
    # Si no se obtienen descriptores, devolvemos un vector de características neutro.
    if descriptor is None or len(descriptor) == 0:
        return [0, 0, 0]
    
    # Calcular estadísticas del descriptor HOG: media, desviación estándar y valor máximo.
    mean_val  = np.mean(descriptor)
    std_val   = np.std(descriptor)
    max_val   = np.max(descriptor)
    
    return [mean_val, std_val, max_val]
