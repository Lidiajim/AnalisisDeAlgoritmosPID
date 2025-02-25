import cv2
import numpy as np
import xml.etree.ElementTree as ET

def extraer_harris_features(image, block_size=2, ksize=3, k=0.04):
    """Extrae características estadísticas a partir de Harris Corner."""
    corners = cv2.cornerHarris(image, block_size, ksize, k)
    corners = cv2.dilate(corners, None)  # Destacar esquinas
    
    # Extraer estadísticas de las esquinas (media, desviación, máximo)
    mean = np.mean(corners)
    std = np.std(corners)
    max_val = np.max(corners)

    return [mean, std, max_val]

import cv2
import numpy as np
import xml.etree.ElementTree as ET

def extraer_harris_features(image, block_size=2, ksize=3, k=0.04):
    """Extrae características estadísticas a partir de Harris Corner usando OpenCV."""
    corners = cv2.cornerHarris(image, block_size, ksize, k)
    corners = cv2.dilate(corners, None)  # Destacar esquinas
    
    # Extraer estadísticas de las esquinas (media, desviación, máximo)
    mean = np.mean(corners)
    std = np.std(corners)
    max_val = np.max(corners)

    return [mean, std, max_val]

def extraer_harris_custom_features(image):
    """Extrae características estadísticas a partir de Harris Corner usando el algoritmo personalizado."""
    # Se calcula el gradiente de x e y
    Ix = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    # Se calcula los productos de los gradientes
    Ix2 = Ix * Ix
    Iy2 = Iy * Iy
    Ixy = Ix * Iy

    # Se aplica un filtro Gaussiano para suavizar
    Ix2 = cv2.GaussianBlur(Ix2, (3, 3), 1)
    Iy2 = cv2.GaussianBlur(Iy2, (3, 3), 1)
    Ixy = cv2.GaussianBlur(Ixy, (3, 3), 1)

    # Calculamos el Harris Score
    k = 0.04
    detM = (Ix2 * Iy2) - (Ixy ** 2)
    traceM = Ix2 + Iy2
    R = detM - k * (traceM ** 2)

    # Normalizamos y aplicamos un umbral
    R_norm = cv2.normalize(R, None, 0, 255, cv2.NORM_MINMAX)
    R_norm = np.uint8(R_norm)
    threshold = 100
    corners = R_norm > threshold

    # Aplicamos supresión de no máximos
    window_size = 5
    dilated = cv2.dilate(R, np.ones((window_size, window_size), np.uint8))  # Máximo local
    strongest_corners = (R == dilated) & corners  # Solo conservar los valores máximos

    mean = np.mean(strongest_corners)
    std = np.std(strongest_corners)
    max_val = np.max(strongest_corners)

    return [mean, std, max_val]

def procesar_imagen(image_path, xml_path, use_custom_harris=False):
    """ Extrae características de Harris de la imagen con anotaciones en XML """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: No se pudo cargar la imagen {image_path}")
        return None, None
    
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    X, y = [], []
    for obj in root.findall("object"):
        label = obj.find("name").text.lower()
        if label == "persona":
            bbox = obj.find("bndbox")
            xmin, ymin = int(bbox.find("xmin").text), int(bbox.find("ymin").text)
            xmax, ymax = int(bbox.find("xmax").text), int(bbox.find("ymax").text)
            
            # Extraer región de interés
            roi = image[ymin:ymax, xmin:xmax]
            features = extraer_harris_custom_features(roi) if use_custom_harris else extraer_harris_features(roi)
            
            if features:
                print(f"Características extraídas: {features}")
                X.append(features)
                y.append(1)  # Etiqueta 1 para persona
    
    if not X:
        print(f"No se encontraron personas en la imagen {image_path}")
    
    return X, y
