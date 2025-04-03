import os
import cv2
import joblib
import numpy as np
import xml.etree.ElementTree as ET
import random as rn
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

class svm:
    def __init__(self, algorithm, algo_params):
        self.algorithm = algorithm
        self.algo_params = algo_params
        self.model = SVC(kernel="rbf", C=1)
    
    def extract_features(self, image, roi=None):
        region = roi if roi is not None else image
        return self.algorithm(region, self.algo_params)
    
    def generateRndRoi(self, image):
            
        maxTamx = int((image.shape[1]*0.3)) #El roi debe ser máximo el 30% de la imagen
        minTamx = int((image.shape[1]*0.15)) #El roi debe ser superior al 15% de la imagen
        maxTamy = int((image.shape[0]*0.3)) #El roi debe ser máximo el 30% de la imagen
        minTamy = int((image.shape[0]*0.15)) #El roi debe ser superior al 15% de la imagen
        
        width = rn.randint(minTamy, maxTamy)
        height = rn.randint(minTamx, maxTamx)

        '''1º. Calculamos el area maxima del roi, y su area minima en base al tamaño de la imagen.
           2º. Generamos un area aleatoria entre esos extremos
           3º. Generamos una altura aleatoria teniendo en cuenta que debe ser como minimo 1/4 y como maximo 1/2 del tamaño del area
           4º. Generamos la anchura correspodiente al area y la altura
         '''
        #Ahora hay que generar un punto que va a ser el ancla de nuestra roi
        #Para ello, debemos asegurarnos que nuestra roi siempre este dentro de la imagem

        xmin = rn.randint(0, (image.shape[1] - width)) #Calcula una x entre [0, width max de la imagen]
        ymin = rn.randint(0, (image.shape[0] - height)) #Calcula una y entre [0, height max de la imagen]
        xmax = xmin + width
        ymax = ymin + height
        roi = image[ymin:ymax, xmin:xmax]
        #print(f"ROI size: ({roi.shape[0]},{roi.shape[1]}) (ymin: {ymin}, ymax: {ymax}, xmin: {xmin}, xmax: {xmax})")
        return roi

    def process_image(self, image_path, xml_path):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return [], []
        
        try:
            tree = ET.parse(xml_path)
        except:
            return [], []
        
        root = tree.getroot()
        features_list, labels_list = [], []
        for obj in root.findall("object"):
            if obj.find("name").text.lower() == "persona":
                bbox = obj.find("bndbox")
                xmin, ymin = int(bbox.find("xmin").text), int(bbox.find("ymin").text)
                xmax, ymax = int(bbox.find("xmax").text), int(bbox.find("ymax").text)
                roi = image[ymin:ymax, xmin:xmax]
                features = self.extract_features(image, roi)
                if features is not None:
                    features_list.append(features)
                    labels_list.append(1)
        return features_list, labels_list
    
    def load_data_persona(self, images_dir):
        X, y = [], []
        for file in os.listdir(images_dir):
            if file.endswith(".jpg"):
                xml_path = os.path.join(images_dir, file.replace(".jpg", ".xml"))
                if os.path.exists(xml_path):
                    feats, labels = self.process_image(os.path.join(images_dir, file), xml_path)
                    X.extend(feats)
                    y.extend(labels)
        return np.array(X), np.array(y)
    
    def load_data_no_persona(self, images_dir):
        X, y = [], []
        for file in os.listdir(images_dir):
            if file.endswith(".jpg"):
                image = cv2.imread(os.path.join(images_dir, file), cv2.IMREAD_GRAYSCALE)
                if image is not None:
                    roi = self.generateRndRoi(image)
                    features = self.extract_features(image, roi)
                    if features is not None:
                        X.append(features)
                        y.append(0)
        return np.array(X), np.array(y)
    
   
    
    def fit(self, train_persona_dir, train_no_persona_dir):
        X_train_p, y_train_p = self.load_data_persona(train_persona_dir)
        X_train_n, y_train_n = self.load_data_no_persona(train_no_persona_dir)
        
        print(f"Imágenes procesadas - Entrenamiento: {len(y_train_p)} personas, {len(y_train_n)} sin persona")
        
        X_train = np.vstack((X_train_p, X_train_n))
        y_train = np.concatenate((y_train_p, y_train_n))
        
        if len(np.unique(y_train)) < 2:
            raise ValueError("El dataset debe contener al menos dos clases diferentes para entrenar un SVM.")
        
        self.model.fit(X_train, y_train)
        
        #Codigo no usado

        #if val_persona_dir and val_no_persona_dir:
        #    X_val_p, y_val_p = self.load_data_persona(val_persona_dir)
        #    X_val_n, y_val_n = self.load_data_no_persona(val_no_persona_dir)
        #    print(f"Imágenes procesadas - Validación: {len(y_val_p)} personas, {len(y_val_n)} sin persona")
    
    def predict_persona(self, images_dir):
        X, y_true = [], []
        for file in os.listdir(images_dir):
            if file.endswith(".jpg"):
                xml_path = os.path.join(images_dir, file.replace(".jpg", ".xml"))
                if os.path.exists(xml_path):
                    feats, labels = self.process_image(os.path.join(images_dir, file), xml_path)
                    X.extend(feats)
                    y_true.extend(labels)
        return np.array(y_true), np.array(self.model.predict(np.array(X))) if X else (np.array([]), np.array([]))
    
    def predict_no_persona(self, images_dir):
        X, y_true = [], []
        for file in os.listdir(images_dir):
            if file.endswith(".jpg"):
                image = cv2.imread(os.path.join(images_dir, file), cv2.IMREAD_GRAYSCALE)
                if image is not None:
                    roi = self.generateRndRoi(image)
                    features = self.extract_features(image, roi)
                    if features is not None:
                        X.append(features)
                        y_true.append(0)
        return np.array(y_true), np.array(self.model.predict(np.array(X))) if X else (np.array([]), np.array([]))
   
    def evaluate(self, test_persona_dir, test_no_persona_dir, model_path=None):
        if model_path:
            self.model = joblib.load(model_path)
        
        y_true_p, y_pred_p = self.predict_persona(test_persona_dir)
        y_true_np, y_pred_np = self.predict_no_persona(test_no_persona_dir)
        
        y_true = np.concatenate((y_true_p, y_true_np))
        y_pred = np.concatenate((y_pred_p, y_pred_np)) if y_pred_p.size and y_pred_np.size else y_pred_p if y_pred_p.size else y_pred_np

        print(f"Imágenes procesadas - Evaluacion: {len(y_true_p)} personas, {len(y_true_np)} sin persona")
        if y_true.size > 0 and y_pred.size > 0:
            # Precisión
            accuracy = accuracy_score(y_true, y_pred) * 100
            print(f"\nPrecisión del modelo en el conjunto de prueba: {accuracy:.2f}%")
            
            # Precisión, Recall y F1-Score
            precision = precision_score(y_true, y_pred) * 100
            recall = recall_score(y_true, y_pred) * 100
            f1 = f1_score(y_true, y_pred) * 100
            
            print(f"🔹 Precisión (Precision): {precision:.2f}%")
            print(f"🔹 Sensibilidad (Recall): {recall:.2f}%")
            print(f"🔹 F1-Score: {f1:.2f}%")
            
            # Matriz de confusión
            cm = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = cm.ravel()
            
            print(f"\nMatriz de confusión:")
            print(cm)
            
            # Falsos positivos y falsos negativos
            print(f"\nFalsos positivos (FP): {fp}")
            print(f"Falsos negativos (FN): {fn}")
            print(f"Verdaderos positivos (TP): {tp}")
            print(f"Verdaderos negativos (TN): {tn}")
            
            # Reporte de clasificación
            print("\nReporte de clasificación:")
            report = classification_report(y_true, y_pred, target_names=['No Persona', 'Persona'])
            print(report)
            
        else:
            print("\nNo se encontraron datos para evaluar.")

    
    def save_model(self, filename=None):
        """
        Guarda el modelo entrenado. Si no se proporciona 'filename', se usa un nombre
        por defecto basado en el algoritmo (por ejemplo, 'svm_sift.pkl' para 'sift_svm').
        """
        if filename is None:
            # Se espera que el nombre de la función sea algo como "<nombre>_svm"
            base = self.algorithm.__name__.replace("_svm", "")
            filename = f"svm_{base}.pkl"
        joblib.dump(self.model, filename)
        print(f"Modelo SVM guardado como '{filename}'")


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
