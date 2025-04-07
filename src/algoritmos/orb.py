import cv2
import numpy as np
import matplotlib.pyplot as plt

class ORB_Detector:
    def __init__(self, nfeatures=500):
        """
        Inicializa el detector ORB con un número máximo de características.

        ORB (Oriented FAST and Rotated BRIEF) combina el detector FAST y el descriptor BRIEF,
        agregando rotación y manejo multiescala. Este constructor permite establecer
        el número máximo de keypoints a detectar.
        """
        self.orb = cv2.ORB_create(nfeatures=nfeatures)

    def load_image(self, path):
        """
        Carga una imagen desde un archivo y la convierte a escala de grises.

        ORB trabaja sobre intensidades, por lo tanto, se utiliza la imagen en escala de grises
        para la detección y descripción de características.
        """
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img, gray

    def draw_keypoints(self, img, keypoints):
        """
        Dibuja los keypoints sobre la imagen original usando círculos ricos en información.

        Visualizar los keypoints ayuda a comprender su distribución espacial y orientación.
        """
        return cv2.drawKeypoints(img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    def show_image(self, img, title):
        """
        Muestra una imagen en pantalla con título, usando matplotlib.

        Ideal para visualizar resultados en notebooks o entornos interactivos.
        """
        plt.figure(figsize=(8, 6))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis("off")
        plt.show()

    def compute_harris_response(self, gray, keypoints, threshold=0.01):
        """
        Aplica la respuesta de Harris a los keypoints detectados inicialmente para filtrar los más fuertes.

        ORB mejora la calidad de los puntos FAST aplicando la medida de esquina de Harris,
        seleccionando solo los más robustos según la respuesta de intensidad.
        """
        harris = cv2.cornerHarris(gray, 2, 3, 0.04)
        filtered_keypoints = [kp for kp in keypoints if harris[int(kp.pt[1]), int(kp.pt[0])] > threshold * harris.max()]
        return filtered_keypoints

    def assign_orientation(self, keypoints, gray):
        """
        Asigna orientación a cada keypoint basado en el momento del parche centrado en él.

        Como FAST no proporciona orientación, ORB calcula una dirección dominante basada en
        el centroide ponderado por intensidad. Esto hace que los descriptores sean invariantes a la rotación.
        """
        for kp in keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            patch_size = 31
            x0, y0 = max(x - patch_size // 2, 0), max(y - patch_size // 2, 0)
            x1, y1 = min(x + patch_size // 2, gray.shape[1] - 1), min(y + patch_size // 2, gray.shape[0] - 1)
            patch = gray[y0:y1, x0:x1]
            if patch.size == 0:
                continue
            m = cv2.moments(patch)
            if m["m00"] != 0:
                cx = m["m10"] / m["m00"]
                cy = m["m01"] / m["m00"]
                angle = np.degrees(np.arctan2(cy - patch.shape[0] / 2, cx - patch.shape[1] / 2))
                kp.angle = angle
        return keypoints

    def compute_descriptors(self, gray, keypoints):
        """
        Calcula los descriptores BRIEF rotados (rBRIEF) para los keypoints dados en la imagen gris.

        ORB mejora BRIEF aplicando una rotación según la orientación del keypoint.
        Esto permite una descripción robusta frente a rotaciones.
        """
        return self.orb.compute(gray, keypoints)

    def build_image_pyramid(self, image, n_levels=4, scale_factor=1.2):
        """
        Construye una pirámide de imágenes reduciendo el tamaño en cada nivel.

        :param image: Imagen base (en escala de grises)
        :param n_levels: Número de niveles de la pirámide
        :param scale_factor: Factor de reducción por nivel
        :return: Lista de imágenes a diferentes escalas
        """
        pyramid = [image]
        current = image.copy()

        for _ in range(1, n_levels):
            h, w = current.shape[:2]
            new_size = (int(w / scale_factor), int(h / scale_factor))
            if new_size[0] < 16 or new_size[1] < 16:  # evitar imágenes muy pequeñas
                break
            resized = cv2.resize(current, new_size, interpolation=cv2.INTER_LINEAR)
            pyramid.append(resized)
            current = resized

        return pyramid


    def match_descriptors(self, desc1, desc2):
        """
        Realiza la comparación entre dos conjuntos de descriptores usando BFMatcher con distancia de Hamming.

        Los descriptores de ORB son binarios, por lo que se utiliza la distancia de Hamming para comparar y encontrar coincidencias.
        """
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        return bf.match(desc1, desc2)

    def extract_orb_features(self, image, params=None, tipo=None):
        """
        Extrae descriptores ORB de una imagen o región y devuelve un vector resumen.

        Este vector (por ejemplo, la media de todos los descriptores) es útil para modelos
        de machine learning como SVM, que requieren vectores fijos para entrenar y clasificar.
        """
        orb = cv2.ORB_create(**params) if params else self.orb
        keypoints, descriptors = orb.detectAndCompute(image, None)
        if descriptors is not None:
            return np.mean(descriptors, axis=0)
        return None
