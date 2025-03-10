import cv2
import numpy as np
import xml.etree.ElementTree as ET
import harris_svm as hs
import hog_svm as ho
import sift_svm as st
import surf_svm as sf

def procesar_imagen(image_path, xml_path, algoritmo = 1): # 1: Harris, 2: HOG, 3: SIFT, 4: SURF
    """ 
    Metodo encargado de preprocesar la imagen con anotaciones XML, 
    aplicando el algoritmo deseado.
    """
    
    #prueba
    gb=3
    k=0.04
    threshold=0.05
    ws=5

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

            if(algoritmo == 1):
                features = hs.harris_svm(image, gb, k, threshold, ws) #modificar, necesita constructor
            elif(algoritmo == 2):
                features = ho.hog_svm(roi) #modificar, necesita constructor
            elif(algoritmo == 3):
                features = st.sift_svm(roi) #modificar, necesita constructor
            else:
                features = sf.surf_svm(roi) #modificar, necesita constructor
            
            
            if features:
                print(f"Características extraídas: {features}")
                X.append(features)
                y.append(1)  # Etiqueta 1 para persona
    
    if not X:
        print(f"No se encontraron personas en la imagen {image_path}")
    
    return X, y
