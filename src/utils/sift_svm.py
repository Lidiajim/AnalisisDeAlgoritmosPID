import numpy as np

def sift_svm(image, detector, tipo):
    '''
    image -> imagen a procesar
    detector -> objeto SIFT ya creado con nfeatures=X
    tipo -> indica el tipo de vector de características a extraer
    
       Tipos disponibles:
        1. Estadísticas globales (media, std, max)
        2. Histograma de magnitudes de los keypoints
    
    '''

    keypoints, descriptors = detector.detectAndCompute(image, None)
    # Tipo 1 – Estadísticas: media, std, max
    if tipo == 1:
        if descriptors is None or descriptors.shape[0] == 0:
            return np.zeros(128 * 3)

        mean_desc = np.mean(descriptors, axis=0)
        std_desc = np.std(descriptors, axis=0)
        max_desc = np.max(descriptors, axis=0)

        return np.concatenate([mean_desc, std_desc, max_desc])

    # Tipo 2 – Histograma de magnitudes
    elif tipo == 2:
        if keypoints is None or len(keypoints) == 0:
            return np.zeros(10)

        magnitudes = np.array([kp.response for kp in keypoints])
        hist, _ = np.histogram(magnitudes, bins=10, range=(0, 1))
        return hist.astype(np.float32)

    else:
        raise ValueError(f"Tipo de extracción no válido: {tipo}")
