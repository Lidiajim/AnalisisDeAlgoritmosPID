import cv2
import numpy as np
import algoritmos.harris as hs

def harris_svm(image, parametros):
    
    '''
    image -> imagen a procesar
    gb -> tamaño del filtro gaussiano
    k -> parametro de control del harris score
    ws -> tamaño de la ventana de supresion de no maximos

    ########################################## PASOS DEL ALGORITMO ##########################################
    
        1. Calculamos gradiente de x e y.
        2. Calculamos producto de los gradiente.
        3. Aplicamos un filtro Gaussiano.
        4. Calculamos el Harris Score.
        5. Normalizamos y aplicamos un umbral.
        6. Aplicamos supresión de no maximos.

    El objetivo de este código es facilitarle al SVM un método para la extracción
    de características aplicando el algoritmo Harris Corner Detection.
    '''

    #Inicializamos el constructor
    hs_svm = hs.harris_detect(
        gaussbox = parametros.get("gaussbox", 3),
        k = parametros.get("k", 0.04),
        threshold = parametros.get("threshold", 0.1),
        window_size = parametros.get("window_size", 5)
    )
    
    # Se calcula el gradiente de x e y
    Ix, Iy = hs_svm.calc_grad(image)

    # Se calcula los productos de los gradientes
    Ix2, Iy2, Ixy = hs_svm.calc_grad_prod(Ix, Iy)
    
    # Se aplica un filtro Gaussiano para suavizar
    Ix2, Iy2, Ixy = hs_svm.gauss_filter(Ix2, Iy2, Ixy)

    # Calculamos el Harris Score
    R = hs_svm.calc_harris_score(Ix2, Iy2, Ixy)

    # Normalizamos y aplicamos un umbral
    R_norm = hs_svm.normalize_thresh(R)

    # Aplicamos supresión de no máximos
    strongest_corners = hs_svm.non_max_supre(R_norm)

    mean = np.mean(strongest_corners)
    std = np.std(strongest_corners)
    max_val = np.max(strongest_corners)

    return [mean, std, max_val]

