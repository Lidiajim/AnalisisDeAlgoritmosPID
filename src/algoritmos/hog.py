import cv2
import matplotlib.pyplot as plt
import numpy as np

'''
    Clase que implementa el algoritmo HOG (Histogram of Oriented Gradients)
    para la detección de humanos.

    ########################################## PARÁMETROS ##########################################
    
    cell_size: 
            Type: tuple (int, int)
            Description: Tamaño en píxeles de cada celda (por ejemplo, (8,8)).

    block_size: 
            Type: tuple (int, int)
            Description: Tamaño del bloque en número de celdas (por ejemplo, (2,2)). 
                         El bloque se usa para la normalización.

    block_stride: 
            Type: tuple (int, int)
            Description: Desplazamiento del bloque en píxeles (por ejemplo, (8,8)).

    nbins: 
            Type: int
            Description: Número de bins del histograma de orientaciones (por ejemplo, 9).

    win_stride:
            Type: tuple (int, int)
            Description: Paso de la ventana deslizante para la detección (por ejemplo, (8,8)).

    padding:
            Type: tuple (int, int)
            Description: Tamaño del padding que se aplica a la imagen (por ejemplo, (16,16)).

    scale:
            Type: float
            Description: Factor de escala entre ventanas (por ejemplo, 1.05).

    ########################################## MÉTODOS ##########################################

    compute_hog(img):
        Return: descriptor
        Param: img -> Matriz de la imagen.
        Calcula el descriptor HOG de la imagen usando el HOGDescriptor de OpenCV.

    detect_humans(img, threshold=0.5):
        Return: filtered_rects, filtered_weights
        Param: 
            img -> Matriz de la imagen en formato BGR.
            threshold -> Umbral de confianza para filtrar las detecciones.
        Aplica la detección de humanos utilizando una ventana deslizante con el HOG y un SVM preentrenado,
        utilizando diferentes escalas de la imagen.

    draw_detections(img, rects):
        Return: imagen con las detecciones dibujadas.
        Param: 
            img -> Matriz de la imagen original.
            rects -> Lista de rectángulos (x, y, w, h) donde se detectaron humanos.
        Dibuja rectángulos sobre la imagen en las posiciones detectadas.

    draw_img(img, axis="on"):
        Param: 
            img -> Matriz de la imagen.
            axis -> Controla si se muestran los ejes.
        Muestra la imagen con matplotlib.
'''

class hog_detect:
    def __init__(self, cell_size=(8,8), block_size=(2,2), block_stride=(8,8), nbins=9,
                 win_stride=(8,8), padding=(16,16), scale=1.05):
        # Inicialización de parámetros
        self.cell_size = cell_size
        self.block_size = block_size
        self.block_stride = block_stride
        self.nbins = nbins
        self.win_stride = win_stride
        self.padding = padding
        self.scale = scale
        
        # Definir el tamaño de la ventana de detección
        # Se suele usar el tamaño (64, 128) para detección de peatones
        winSize = (64, 128)
        blockSize = (self.block_size[0]*self.cell_size[0],
                     self.block_size[1]*self.cell_size[1])
        
        # Inicializar el descriptor HOG de OpenCV con los parámetros dados
        self.hog = cv2.HOGDescriptor(_winSize=winSize,
                                     _blockSize=blockSize,
                                     _blockStride=self.block_stride,
                                     _cellSize=self.cell_size,
                                     _nbins=self.nbins)
        
        # Cargar el detector SVM preentrenado para personas
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def to_grayscale(self, img):
        """
        Convierte la imagen a escala de grises.
        
        Input:
            img: Imagen en formato BGR o en escala de grises.
        Return:
            gray: Imagen en escala de grises.
        """
        if len(img.shape) > 2:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        return gray

    def compute_gradients(self, img):
        """
        Calcula los gradientes de la imagen usando el filtro Sobel.
        
        Input:
            img: Imagen en escala de grises.
        Return:
            magnitude: Matriz de magnitudes del gradiente.
            orientation: Matriz de orientaciones del gradiente en grados (rango [0, 180)).
        """
        Gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
        Gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = np.sqrt(Gx**2 + Gy**2)
        orientation = np.arctan2(Gy, Gx) * (180 / np.pi)
        orientation[orientation < 0] += 180  # Ajustar a rango [0, 180)
        return magnitude, orientation

    def cell_histogram(self, cell_mag, cell_ori):
        """
        Calcula el histograma de orientaciones para una celda.
        
        Input:
            cell_mag: Magnitudes de los gradientes en la celda.
            cell_ori: Orientaciones de los gradientes en la celda.
        Return:
            hist: Histograma de orientaciones para la celda.
        """
        hist = np.zeros(self.nbins)
        h, w = cell_mag.shape
        for i in range(h):
            for j in range(w):
                mag = cell_mag[i, j]
                angle = cell_ori[i, j]
                bin_idx = int(angle // self.bin_width) % self.nbins
                hist[bin_idx] += mag
        return hist

    def compute_cell_histograms(self, magnitude, orientation):
        """
        Divide la imagen en celdas y calcula el histograma en cada una.
        
        Input:
            magnitude: Matriz de magnitudes del gradiente.
            orientation: Matriz de orientaciones del gradiente.
        Return:
            cell_hists: Array con el histograma de cada celda.
        """
        cell_h, cell_w = self.cell_size
        h, w = magnitude.shape
        n_cells_y = h // cell_h
        n_cells_x = w // cell_w
        cell_hists = np.zeros((n_cells_y, n_cells_x, self.nbins))
        for i in range(n_cells_y):
            for j in range(n_cells_x):
                cell_mag = magnitude[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
                cell_ori = orientation[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
                cell_hists[i, j, :] = self.cell_histogram(cell_mag, cell_ori)
        return cell_hists

    def normalize_blocks(self, cell_hists):
        """
        Agrupa las celdas en bloques y normaliza los histogramas de cada bloque.
        
        Input:
            cell_hists: Array de histogramas de celdas.
        Return:
            hog_vector: Vector final del descriptor HOG.
        """
        block_h, block_w = self.block_size  # en número de celdas
        n_cells_y, n_cells_x, _ = cell_hists.shape
        n_blocks_y = n_cells_y - block_h + 1
        n_blocks_x = n_cells_x - block_w + 1
        hog_vector = []
        for i in range(n_blocks_y):
            for j in range(n_blocks_x):
                # Extraer el bloque de celdas
                block = cell_hists[i:i+block_h, j:j+block_w, :].flatten()
                # Normalización L2
                norm = np.linalg.norm(block) + 1e-6  # evitar división por cero
                block_norm = block / norm
                hog_vector.extend(block_norm)
        return np.array(hog_vector)

    def compute_hog(self, img):
        """
        Calcula el descriptor HOG completo de una imagen.
        
        Input:
            img: Imagen original (BGR o escala de grises).
        Return:
            hog_descriptor: Vector final del descriptor HOG.
        """
        gray = self.to_grayscale(img)
        magnitude, orientation = self.compute_gradients(gray)
        cell_hists = self.compute_cell_histograms(magnitude, orientation)
        hog_descriptor = self.normalize_blocks(cell_hists)
        return hog_descriptor
    
    def detect_humans(self, img, threshold=0.5):
        """
        Detecta humanos en la imagen usando el descriptor HOG y un SVM preentrenado.
        Filtra las detecciones por un umbral de confianza (threshold).

        Input:
            img -> Matriz de la imagen en formato BGR.
            threshold -> Umbral de confianza para filtrar las detecciones.
        Return:
            filtered_rects -> Lista de rectángulos (x, y, w, h) de las detecciones filtradas.
            filtered_weights -> Lista de puntajes/confianza correspondientes a cada detección.
        """
        rects, weights = self.hog.detectMultiScale(
            img,
            winStride=self.win_stride,
            padding=self.padding,
            scale=self.scale
            
        )
        
        # Filtrar detecciones con un peso por encima del threshold
        filtered_rects = []
        filtered_weights = []
        for (rect, w) in zip(rects, weights):
            if w > threshold:
                filtered_rects.append(rect)
                filtered_weights.append(w)

        return filtered_rects, filtered_weights

    def draw_detections(self, img, rects):
        """
        Dibuja rectángulos en la imagen en las posiciones detectadas.
        Input:
            img -> Matriz de la imagen original.
            rects -> Lista de detecciones (x, y, w, h).
        Return:
            output_img -> Imagen con rectángulos dibujados.
        """
        output_img = img.copy()
        for (x, y, w, h) in rects:
            cv2.rectangle(output_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        return output_img

    def draw_img(self, img, axis="on"):
        """
        Muestra la imagen usando matplotlib.
        Input:
            img -> Matriz de la imagen (en formato BGR).
            axis -> "on" para mostrar ejes, "off" para ocultarlos.
        """
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title("Detección de Humanos con HOG")
        if axis == "off":
            plt.axis("off")
        plt.show()
