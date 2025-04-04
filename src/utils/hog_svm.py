import numpy as np
import algoritmos.hog as hog

def hog_svm(image, parametros, tipo):
    """
    image -> Imagen a procesar.
    parametros -> Diccionario con los parámetros del detector HOG.
    tipo -> Parámetro adicional para indicar el modo o tipo de procesamiento (puede no ser utilizado directamente).

    ########################################## PASOS DEL ALGORITMO ##########################################
        1. Inicializamos el detector HOG con los parámetros dados.
        2. Calculamos el descriptor HOG para la imagen.
        3. Extraemos estadísticas del descriptor (media, desviación estándar y valor máximo).

    El objetivo es facilitar al SVM un método para la extracción de características aplicando el algoritmo HOG.
    """
    # Inicializamos el detector HOG con los parámetros proporcionados
    hog_detector = hog.HOGDetect(
        cell_size = parametros.get("cell_size", (8, 8)),
        block_size = parametros.get("block_size", (2, 2)),
        block_stride = parametros.get("block_stride", (4, 4)),
        nbins = parametros.get("nbins", 9),
        win_stride = parametros.get("win_stride", (8, 8)),
        padding = parametros.get("padding", (8, 8)),
        scale = parametros.get("scale", 1.05),
        bin_width = parametros.get("bin_width", 20)
    )
    
    # Calcular el descriptor HOG para la imagen
    descriptor = hog_detector.compute_hog(image)
    
    # Si no se obtiene descriptor, se retornan valores neutros
    if descriptor is None or len(descriptor) == 0:
        return [0, 0, 0]
    
    # Extraer estadísticas: media, desviación estándar y máximo del descriptor
    mean_val = np.mean(descriptor)
    std_val  = np.std(descriptor)
    max_val  = np.max(descriptor)
    
    return [mean_val, std_val, max_val]
