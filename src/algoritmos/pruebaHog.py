import cv2
import numpy as np
import matplotlib.pyplot as plt

class hog_detect:
    def __init__(self, cell_size=8, block_size=2, bins=9):
        """
        Parámetros:
            cell_size: Tamaño de cada celda (por ejemplo, 8x8 píxeles).
            block_size: Número de celdas en cada bloque (por ejemplo, 2x2).
            bins: Número de bins para el histograma (por ejemplo, 9, abarcando ángulos de 0 a 180 grados).
        """
        self.cell_size = cell_size
        self.block_size = block_size
        self.bins = bins

    def preprocess(self, image):
        """
        Convierte la imagen a escala de grises si es necesario.
        
        Param:
            image: Imagen de entrada (puede ser a color o en escala de grises).
        Return:
            gray: Imagen en escala de grises.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        return gray

    def compute_gradients(self, gray):
        """
        Calcula los gradientes de la imagen en las direcciones x e y usando Sobel.
        
        Param:
            gray: Imagen en escala de grises.
        Return:
            magnitude: Magnitud del gradiente.
            angle: Ángulo del gradiente (en grados, rango [0,180)).
        """
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=1)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=1)
        magnitude, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
        # Convertir ángulos al rango [0, 180)
        angle = angle % 180
        return magnitude, angle

    def compute_cell_histograms(self, magnitude, angle):
        """
        Divide la imagen en celdas y calcula el histograma de orientaciones en cada una,
        ponderado por la magnitud del gradiente.
        
        Param:
            magnitude: Matriz de magnitudes del gradiente.
            angle: Matriz de ángulos del gradiente.
        Return:
            cell_histograms: Matriz con el histograma de cada celda.
        """
        h, w = magnitude.shape
        n_cells_y = h // self.cell_size
        n_cells_x = w // self.cell_size

        cell_histograms = np.zeros((n_cells_y, n_cells_x, self.bins))
        # Tamaño del intervalo para cada bin
        bin_size = 180 / self.bins

        for i in range(n_cells_y):
            for j in range(n_cells_x):
                # Extraer la región correspondiente a la celda
                cell_mag = magnitude[i*self.cell_size:(i+1)*self.cell_size,
                                     j*self.cell_size:(j+1)*self.cell_size]
                cell_angle = angle[i*self.cell_size:(i+1)*self.cell_size,
                                   j*self.cell_size:(j+1)*self.cell_size]
                # Calcular histograma para la celda con ponderación por magnitud
                hist, _ = np.histogram(cell_angle, bins=self.bins, range=(0, 180), weights=cell_mag)
                cell_histograms[i, j, :] = hist

        return cell_histograms

    def normalize_blocks(self, cell_histograms):
        """
        Realiza la normalización de los histogramas en bloques para obtener invarianza a cambios
        de iluminación y contraste.
        
        Param:
            cell_histograms: Matriz con los histogramas de cada celda.
        Return:
            feature_vector: Vector HOG final concatenado de todos los bloques.
        """
        n_cells_y, n_cells_x, _ = cell_histograms.shape
        normalized_blocks = []

        # Recorrer cada bloque de celdas
        for i in range(n_cells_y - self.block_size + 1):
            for j in range(n_cells_x - self.block_size + 1):
                # Extraer el bloque y aplanarlo en un vector
                block = cell_histograms[i:i+self.block_size, j:j+self.block_size, :].flatten()
                # Normalización L2 con pequeña constante para evitar división por cero
                norm = np.linalg.norm(block) + 1e-6
                normalized_block = block / norm
                normalized_blocks.append(normalized_block)

        # Concatenar todos los bloques para formar el vector HOG final
        feature_vector = np.hstack(normalized_blocks)
        return feature_vector

    def compute_hog(self, image):
        """
        Realiza la extracción del descriptor HOG a partir de la imagen.
        
        Param:
            image: Imagen de entrada (a color o en escala de grises).
        Return:
            feature_vector: Vector final del descriptor HOG.
            cell_histograms: Histograma de cada celda (útil para visualización).
        """
        # 1. Preprocesamiento: convertir a escala de grises
        gray = self.preprocess(image)
        # 2. Cálculo del gradiente: magnitud y dirección
        magnitude, angle = self.compute_gradients(gray)
        # 3. División en celdas y cálculo de histogramas
        cell_histograms = self.compute_cell_histograms(magnitude, angle)
        # 4. Normalización por bloques y concatenación
        feature_vector = self.normalize_blocks(cell_histograms)
        
        return feature_vector, cell_histograms

    def draw_img(self, img, axis="on"):
        """
        Muestra la imagen utilizando matplotlib.
        
        Param:
            img: Imagen a mostrar.
            axis: Controla la visualización de los ejes ('on' o 'off').
        """
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title("Imagen")
        if axis == "off":
            plt.axis("off")
        plt.show()

    def draw_hog_descriptor(self, cell_histograms, scale=5):
        """
        Dibuja una imagen con el descriptor HOG pintado.

        Parámetros:
            cell_histograms: Matriz de histogramas por celda (resultado de compute_cell_histograms).
            scale: Factor de escala para ajustar la longitud de las líneas (puedes ajustar este valor según convenga).

        Retorna:
            hog_image: Imagen en la que se visualiza el descriptor HOG.
        """
        n_cells_y, n_cells_x, bins = cell_histograms.shape
        img_height = n_cells_y * self.cell_size
        img_width = n_cells_x * self.cell_size
        hog_image = np.zeros((img_height, img_width, 3), dtype=np.uint8)

        bin_angle = 180 / bins

        for i in range(n_cells_y):
            for j in range(n_cells_x):
                # Centro de la celda
                center_x = int(j * self.cell_size + self.cell_size / 2)
                center_y = int(i * self.cell_size + self.cell_size / 2)
                # Por cada bin se dibuja una línea
                for bin_idx in range(bins):
                    # Ángulo central del bin (en grados)
                    angle_deg = bin_idx * bin_angle + bin_angle / 2
                    angle_rad = np.deg2rad(angle_deg)
                    # Longitud proporcional al valor del histograma
                    magnitude_value = cell_histograms[i, j, bin_idx]
                    dx = int(magnitude_value * scale * np.cos(angle_rad))
                    dy = int(magnitude_value * scale * np.sin(angle_rad))
                    pt1 = (center_x - dx, center_y - dy)
                    pt2 = (center_x + dx, center_y + dy)
                    cv2.line(hog_image, pt1, pt2, (255, 255, 255), 1)
        return hog_image






