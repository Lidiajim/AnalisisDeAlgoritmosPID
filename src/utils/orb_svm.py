import algoritmos.orb as orb
import cv2
import matplotlib.pyplot as plt
import numpy as np

def orb_svm(image, parametros, tipo):
    """
    image -> Imagen en escala de grises.
    parametros -> Diccionario con los parámetros de ORB.
    tipo -> No se usa, pero se incluye para mantener compatibilidad con la interfaz.

    1. Inicializa ORB con los parámetros dados.
    2. Extrae los descriptores ORB de la imagen.
    3. Devuelve un vector de características resumen (media, std, max) si hay descriptores válidos.
    """
    orb = cv2.ORB_create(**parametros) if parametros else cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(image, None)

    if descriptors is None or len(descriptors) == 0:
        return [0, 0, 0]  # vector neutro si no hay descriptores

    if tipo == 1:
        # Estadísticas: media, std, max
        mean_val = np.mean(descriptors)
        std_val = np.std(descriptors)
        max_val = np.max(descriptors)
        return [mean_val, std_val, max_val]
