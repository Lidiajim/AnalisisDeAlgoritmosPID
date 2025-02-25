import os
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
from extraer_caracteristicas import procesar_imagen, extraer_harris_features, extraer_harris_custom_features
import cv2

# Rutas del dataset
train_path_persona = "train"
val_path_persona = "valid"
annotations_train_path = train_path_persona
annotations_val_path = val_path_persona

train_path_no_persona = "train_no/scene"
val_path_no_persona = "valid_no/scene"

def cargar_datos_persona(imagenes_path, anotaciones_path):
    """Carga imágenes con personas usando anotaciones XML"""
    X, y = [], []

    for image_file in os.listdir(imagenes_path):
        if image_file.endswith(".jpg"):
            image_path = os.path.join(imagenes_path, image_file)
            xml_path = os.path.join(anotaciones_path, image_file.replace(".jpg", ".xml"))

            if os.path.exists(xml_path):
                print(f"Procesando persona: {image_file}")
                features, labels = procesar_imagen(image_path, xml_path)

                if features:
                    X.extend(features)
                    y.extend(labels)

    return np.array(X), np.array(y)

def cargar_datos_no_persona(imagenes_path):
    """Carga imágenes sin personas y extrae características de toda la imagen"""
    X, y = [], []

    for image_file in os.listdir(imagenes_path):
        if image_file.endswith(".jpg"):
            image_path = os.path.join(imagenes_path, image_file)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            if image is not None:
                print(f"Procesando NO persona: {image_file}")
                features = extraer_harris_features(image)

                if features:
                    X.append(features)
                    y.append(0)  # Etiqueta 0 para "no persona"

    return np.array(X), np.array(y)

# Cargar datos de entrenamiento y validación
X_train_p, y_train_p = cargar_datos_persona(train_path_persona, annotations_train_path)
X_train_n, y_train_n = cargar_datos_no_persona(train_path_no_persona)

X_val_p, y_val_p = cargar_datos_persona(val_path_persona, annotations_val_path)
X_val_n, y_val_n = cargar_datos_no_persona(val_path_no_persona)

# Combinar datos
X_train = np.vstack((X_train_p, X_train_n))
y_train = np.concatenate((y_train_p, y_train_n))

X_val = np.vstack((X_val_p, X_val_n))
y_val = np.concatenate((y_val_p, y_val_n))

print(f"Características de entrenamiento: {X_train.shape}")
print(f"Características de validación: {X_val.shape}")

# Verificar que haya más de una clase
if len(set(y_train)) < 2:
    raise ValueError("El dataset debe contener al menos dos clases diferentes para entrenar un SVM.")

# Entrenar el modelo SVM
svm = SVC(kernel="linear")
svm.fit(X_train, y_train)

# Evaluar con el conjunto de validación
y_pred = svm.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f"Precisión en validación: {accuracy * 100:.2f}%")

# Guardar el modelo entrenado
joblib.dump(svm, "svm_harris.pkl")
print("Modelo SVM guardado como 'svm_harris.pkl'")
