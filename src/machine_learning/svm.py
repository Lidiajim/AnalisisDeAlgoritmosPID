import os
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
from procesar_imagen import procesar_imagen
import harris_svm as hs
import hog_svm as ho
import sift_svm as st
import surf_svm as sf
import cv2

# Rutas del dataset
train_path_persona = "images/train"
val_path_persona = "images/valid"
annotations_train_path = train_path_persona
annotations_val_path = val_path_persona

train_path_no_persona = "images/train_no"
val_path_no_persona = "images/valid_no"
algo_control = 1

#prueba
gb=3
k=0.04
threshold=0.05
ws=5

def cargar_datos_persona(imagenes_path, anotaciones_path, algoritmo=1): # 1: Harris, 2: HOG, 3: SIFT, 4: SURF
    """Carga imágenes con personas usando anotaciones XML"""
    X, y = [], []
    algo_control = algoritmo

    for image_file in os.listdir(imagenes_path):
        if image_file.endswith(".jpg"):
            image_path = os.path.join(imagenes_path, image_file)
            xml_path = os.path.join(anotaciones_path, image_file.replace(".jpg", ".xml"))

            if os.path.exists(xml_path):
                print(f"Procesando persona: {image_file}")
                features, labels = procesar_imagen(image_path, xml_path,algoritmo) #modificar, necesita constructor

                if features:
                    X.extend(features)
                    y.extend(labels)

    return np.array(X), np.array(y)

def cargar_datos_no_persona(imagenes_path, algoritmo=1):
    """Carga imágenes sin personas"""
    X, y = [], []
    algo_control = algoritmo

    for image_file in os.listdir(imagenes_path):
        if image_file.endswith(".jpg"):
            image_path = os.path.join(imagenes_path, image_file)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            if image is not None:
                print(f"Procesando NO persona: {image_file}")
                if(algoritmo == 1):
                    features = hs.harris_svm(image, gb, k, threshold, ws) #modificar, necesita constructor
                elif(algoritmo == 2):
                    features = ho.hog_svm(image) #modificar, necesita constructor
                elif(algoritmo == 3):
                    features = st.sift_svm(image) #modificar, necesita constructor
                else:
                    features = sf.surf_svm(image) #modificar, necesita constructor

                if features:
                    X.append(features)
                    y.append(0)  # Etiqueta 0 para "no persona"

    return np.array(X), np.array(y)

# Cargar datos de entrenamiento y validación
X_train_p, y_train_p = cargar_datos_persona(train_path_persona, annotations_train_path,1)
X_train_n, y_train_n = cargar_datos_no_persona(train_path_no_persona,1)

X_val_p, y_val_p = cargar_datos_persona(val_path_persona, annotations_val_path,1)
X_val_n, y_val_n = cargar_datos_no_persona(val_path_no_persona,1)

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

if(algo_control == 1):
    joblib.dump(svm, "svm_harris_hs.pkl")
    print("Modelo SVM guardado como 'svm_harris_hs.pkl'")                      
elif(algo_control == 2):
    joblib.dump(svm, "svm_harris_hog.pkl")
    print("Modelo SVM guardado como 'svm_harris_hs.pkl'")                     
elif(algo_control == 3):
    joblib.dump(svm, "svm_harris_sift.pkl")
    print("Modelo SVM guardado como 'svm_harris_hs.pkl'")          
else:
    joblib.dump(svm, "svm_harris_surf.pkl") 
    print("Modelo SVM guardado como 'svm_harris_hs.pkl'")    


