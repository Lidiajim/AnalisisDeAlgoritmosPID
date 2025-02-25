import os
import cv2
import numpy as np
import joblib
from extraer_caracteristicas import extraer_harris_features, procesar_imagen, extraer_harris_custom_features

# Cargar el modelo entrenado
svm = joblib.load("svm_harris.pkl")

# Rutas de prueba
test_path_persona = "test"  # Contiene imágenes con personas y archivos XML
test_annotations_path = test_path_persona
test_path_no_persona = "test_no/scene"  # Contiene imágenes sin personas

def predecir_en_imagenes_persona(imagenes_path, anotaciones_path):
    
    X, y_true, y_pred = [], [], []

    for image_file in os.listdir(imagenes_path):
        if image_file.endswith(".jpg"):
            image_path = os.path.join(imagenes_path, image_file)
            xml_path = os.path.join(anotaciones_path, image_file.replace(".jpg", ".xml"))

            if os.path.exists(xml_path):
                print(f"Procesando imagen con persona: {image_file}")
                features, labels = procesar_imagen(image_path, xml_path)

                if features:
                    X.extend(features)
                    y_true.extend(labels)  # 1 para persona

    if len(X) > 0:
        X = np.array(X)
        y_pred = svm.predict(X)
    
    return y_true, y_pred

def predecir_en_imagenes_no_persona(imagenes_path):
    
    X, y_true, y_pred = [], [], []

    for image_file in os.listdir(imagenes_path):
        if image_file.endswith(".jpg"):
            image_path = os.path.join(imagenes_path, image_file)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            if image is None:
                print(f"No se pudo cargar la imagen: {image_path}")
                continue

            print(f"Procesando imagen sin persona: {image_file}")
            features = extraer_harris_features(image)

            if features:
                X.append(features)
                y_true.append(0)  # 0 para no persona

    if len(X) > 0:
        X = np.array(X)
        y_pred = svm.predict(X)
    
    return y_true, y_pred

# Obtener predicciones
y_true_persona, y_pred_persona = predecir_en_imagenes_persona(test_path_persona, test_annotations_path)
y_true_no_persona, y_pred_no_persona = predecir_en_imagenes_no_persona(test_path_no_persona)

# Combinar resultados
y_true = np.concatenate((y_true_persona, y_true_no_persona)) if y_true_persona and y_true_no_persona else np.array(y_true_persona + y_true_no_persona)

#y_pred = np.concatenate((y_pred_persona, y_pred_no_persona)) if y_pred_persona and y_pred_no_persona else np.array(y_pred_persona + y_pred_no_persona)
if y_pred_persona.size > 0 and y_pred_no_persona.size > 0:
    y_pred = np.concatenate((y_pred_persona, y_pred_no_persona))
elif y_pred_persona.size > 0:
    y_pred = y_pred_persona
elif y_pred_no_persona.size > 0:
    y_pred = y_pred_no_persona
else:
    y_pred = np.array([])  # Si no hay predicciones, devuelve un array vacío



# Calcular precisión si hay datos
if len(y_true) > 0:
    accuracy = np.mean(y_true == y_pred) * 100
    print(f"\n Precisión del modelo en el conjunto de prueba: {accuracy:.2f}%")
else:
    print("\nNo se encontraron datos para evaluar.")
