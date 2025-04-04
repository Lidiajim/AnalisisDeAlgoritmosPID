import cv2
import matplotlib.pyplot as plt
import numpy as np

class HOGDetect:
    """
    Clase para la detección de humanos utilizando el descriptor HOG.
    
    Incluye métodos para:
      - Calcular el descriptor HOG manualmente (paso a paso).
      - Detectar personas usando el detector preentrenado de OpenCV.
      - Visualizar resultados y etapas intermedias.
    
    Parámetros comunes:
      cell_size: Tamaño de cada celda (ej. (8, 8)).
      block_size: Número de celdas por bloque (ej. (2, 2)).
      block_stride: Desplazamiento entre bloques.
      nbins: Número de bins en el histograma de orientaciones.
      win_stride: Stride de la ventana en detectMultiScale.
      padding: Padding aplicado a la imagen en detectMultiScale.
      scale: Factor de escala en detectMultiScale.
      bin_width: Ancho de cada bin (en grados) para el histograma.
    """
    def __init__(self, cell_size=(8, 8), block_size=(2, 2), block_stride=(8, 8), nbins=9,
                 win_stride=(8, 8), padding=(16, 16), scale=1.05, bin_width=20):
        """
        Inicializa la clase con parámetros de configuración para HOG.
        """
        self.cell_size = cell_size
        self.block_size = block_size
        self.block_stride = block_stride
        self.nbins = nbins
        self.win_stride = win_stride
        self.padding = padding
        self.scale = scale
        self.bin_width = bin_width

        # Tamaño estándar para detección de peatones
        winSize = (64, 128)
        blockSize = (self.block_size[0] * self.cell_size[0],
                     self.block_size[1] * self.cell_size[1])
        
        # Inicialización del descriptor HOG de OpenCV con los parámetros configurados
        self.hog = cv2.HOGDescriptor(_winSize=winSize,
                                     _blockSize=blockSize,
                                     _blockStride=self.block_stride,
                                     _cellSize=self.cell_size,
                                     _nbins=self.nbins)
        

    def to_grayscale(self, img):
        """
        Convierte la imagen a escala de grises si tiene más de un canal.
        
        Parámetros:
          img: Imagen de entrada (np.ndarray). Debe estar correctamente cargada.
          
        Retorna:
          Imagen en escala de grises.
          
        Ejemplo:
          gray = detector.to_grayscale(img)
        """
        if img is None:
            raise ValueError("La imagen de entrada es None.")
        if len(img.shape) > 2:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def compute_gradients(self, img):
        """
        Calcula las derivadas en X e Y usando el operador Sobel y obtiene la magnitud y orientación.
        
        Parámetros:
          img: Imagen en escala de grises.
          
        Retorna:
          - magnitude: Matriz con la magnitud del gradiente.
          - orientation: Matriz con la orientación (en grados) en el rango [0, 180).
        """
        Gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
        Gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = np.sqrt(Gx**2 + Gy**2)
        orientation = np.arctan2(Gy, Gx) * (180 / np.pi)
        orientation[orientation < 0] += 180
        return magnitude, orientation

    def cell_histogram(self, cell_mag: np.ndarray, cell_ori: np.ndarray) -> np.ndarray:
        """
        Calcula el histograma de orientaciones para una celda con interpolación bilineal en el bin.
        
        En lugar de asignar completamente la contribución al bin calculado, se distribuye la contribución
        entre el bin actual y el siguiente según la proximidad al centro del bin.
        
        Parámetros:
        cell_mag: Matriz de magnitudes de la celda.
        cell_ori: Matriz de orientaciones de la celda.
        
        Retorna:
        Histograma (np.ndarray) de longitud 'nbins'.
        """
        hist = np.zeros(self.nbins)
        bin_width = self.bin_width  # por ejemplo, 20° si nbins = 9
        for i in range(cell_mag.shape[0]):
            for j in range(cell_mag.shape[1]):
                mag = cell_mag[i, j]
                angle = cell_ori[i, j]
                # Normalizamos el ángulo en el rango [0, 180)
                angle = angle % 180
                
                # Calcular el índice base del bin
                bin_idx = angle / bin_width
                lower_bin = int(np.floor(bin_idx)) % self.nbins
                upper_bin = (lower_bin + 1) % self.nbins
                # Proporciones de interpolación
                upper_weight = bin_idx - np.floor(bin_idx)
                lower_weight = 1 - upper_weight
                
                hist[lower_bin] += mag * lower_weight
                hist[upper_bin] += mag * upper_weight
        return hist

    def compute_cell_histograms(self, magnitude, orientation):
        """
        Divide la imagen en celdas y calcula el histograma para cada una.
        
        Parámetros:
          magnitude: Matriz de magnitud del gradiente.
          orientation: Matriz de orientación del gradiente.
          
        Retorna:
          Un arreglo 3D con forma (n_cells_y, n_cells_x, nbins).
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
    
    def demonstrate_cell_histogram(self, img, cell_row, cell_col):
        """
        Demuestra el proceso de obtención del histograma para una celda específica.
        
        Flujo:
          1. Convierte la imagen a gris y calcula los gradientes.
          2. Extrae el histograma para la celda en la posición (cell_row, cell_col).
          3. Muestra la imagen de gradiente, la celda y el histograma.
        
        Parámetros:
          img: Imagen de entrada.
          cell_row: Índice de la fila de la celda.
          cell_col: Índice de la columna de la celda.
        """
        gray = self.to_grayscale(img)
        magnitude, orientation = self.compute_gradients(gray)
        cell_hists = self.compute_cell_histograms(magnitude, orientation)
        
        # Extraer el histograma de la celda especificada
        hist = cell_hists[cell_row, cell_col]
        
        # Definir las coordenadas de la celda
        cell_h, cell_w = self.cell_size
        start_row, start_col = cell_row * cell_h, cell_col * cell_w
        cell_img = gray[start_row:start_row+cell_h, start_col:start_col+cell_w]
        
        # Mostrar resultados
        plt.figure(figsize=(15, 4))
        
        plt.subplot(1, 3, 1)
        plt.imshow(gray, cmap='gray')
        plt.title("Imagen en Escala de Grises")
        plt.axis("off")
        
        plt.subplot(1, 3, 2)
        plt.imshow(cell_img, cmap='gray')
        plt.title(f"Celda ({cell_row}, {cell_col})")
        plt.axis("off")
        
        plt.subplot(1, 3, 3)
        plt.bar(range(self.nbins), hist, width=self.bin_width)
        plt.title("Histograma de la Celda")
        plt.xlabel("Orientación")
        plt.ylabel("Magnitud acumulada")
        
        plt.tight_layout()
        plt.show()

    def normalize_blocks(self, cell_hists: np.ndarray) -> np.ndarray:
        """
        Agrupa las celdas en bloques y normaliza cada bloque utilizando la normalización L2-Hys.
        
        L2-Hys consiste en:
        1. Normalización L2 del bloque.
        2. Clipping de los valores (por ejemplo, a 0.2).
        3. Renormalización L2.
        
        Parámetros:
        cell_hists: Arreglo de histogramas de celdas.
        
        Retorna:
        Vector 1D con el descriptor HOG concatenado.
        """
        block_h, block_w = self.block_size
        n_cells_y, n_cells_x, _ = cell_hists.shape
        n_blocks_y = n_cells_y - block_h + 1
        n_blocks_x = n_cells_x - block_w + 1
        hog_vector = []
        for i in range(n_blocks_y):
            for j in range(n_blocks_x):
                block = cell_hists[i:i+block_h, j:j+block_w, :].flatten()
                # Normalización L2
                norm = np.linalg.norm(block) + 1e-6
                block = block / norm
                # Clipping (L2-Hys): limitar valores a 0.2
                block = np.minimum(block, 0.2)
                # Renormalización L2
                norm = np.linalg.norm(block) + 1e-6
                block = block / norm
                hog_vector.extend(block)
        return np.array(hog_vector)

    def compute_hog(self, img):
        """
        Procesa la imagen paso a paso para obtener el descriptor HOG.
        
        Flujo:
          1. Conversión a escala de grises.
          2. Cálculo de gradientes (magnitud y orientación).
          3. Cálculo de histogramas para cada celda.
          4. Normalización de bloques.
          
        Parámetros:
          img: Imagen de entrada.
          
        Retorna:
          Descriptor HOG en forma de vector.
        """
        gray = self.to_grayscale(img)
        magnitude, orientation = self.compute_gradients(gray)
        cell_hists = self.compute_cell_histograms(magnitude, orientation)
        hog_descriptor = self.normalize_blocks(cell_hists)
        return hog_descriptor

    def detect_humans(self, img, threshold):
        """
        Detección estándar de humanos utilizando el detector preentrenado de HOG en OpenCV.
        
        Se aplican los parámetros definidos (win_stride, padding, scale) y se filtran las detecciones
        según un umbral de confianza.
        
        Parámetros:
          img: Imagen de entrada.
          threshold: Valor mínimo de confianza para aceptar una detección.
          
        Retorna:
          Tuple (filtered_rects, filtered_weights).
        """
        if img is None:
            raise ValueError("La imagen de entrada es None.")
        rects, weights = self.hog.detectMultiScale(
            img,
            winStride=self.win_stride,
            padding=self.padding,
            scale=self.scale
        )
        filtered_rects = []
        filtered_weights = []
        for rect, w in zip(rects, weights):
            if w > threshold:
                filtered_rects.append(rect)
                filtered_weights.append(w)
        return filtered_rects, filtered_weights

    def detect_humans_individual(self, img, threshold = 0.5, sub_threshold = 0.5):
        """
        Detecta humanos intentando separar aquellos que están muy juntos.
        
        Flujo:
          1. Se realiza una detección inicial.
          2. Si un rectángulo detectado es muy ancho (p.ej., > 60 px), se extrae la subregión y se aplica una detección con parámetros más finos.
          3. Se ajustan las coordenadas de las detecciones en la subregión al sistema de coordenadas original.
          
        Parámetros:
          img: Imagen de entrada.
          threshold: Umbral para la detección inicial.
          sub_threshold: Umbral para la detección en la subregión.
          
        Retorna:
          Tuple (final_rects, final_weights).
        """
        rects, weights = self.detect_humans(img, threshold=threshold)
        final_rects = []
        final_weights = []
        for rect, w in zip(rects, weights):
            x, y, ancho_rect, alto_rect = rect
            if ancho_rect > 60:
                subROI = img[y:y + alto_rect, x:x + ancho_rect]
                sub_rects, sub_weights = self.hog.detectMultiScale(
                    subROI,
                    winStride=(2, 2),
                    padding=self.padding,
                    scale=1.005
                )
                for srect, sw in zip(sub_rects, sub_weights):
                    if sw > sub_threshold:
                        sx, sy, sw_rect, sh_rect = srect
                        final_rects.append((x + sx, y + sy, sw_rect, sh_rect))
                        final_weights.append(sw)
                # Si no se detectó nada en la subregión, se conserva el rectángulo original.
                if sub_rects is None or sub_rects.size == 0:
                    final_rects.append(rect)
                    final_weights.append(w)
            else:
                final_rects.append(rect)
                final_weights.append(w)
        return final_rects, final_weights

    def draw_detections(self, img, rects):
        """
        Dibuja los rectángulos de detección sobre la imagen.
        
        Parámetros:
          img: Imagen original.
          rects: Lista de rectángulos de detección.
          
        Retorna:
          Imagen con los rectángulos dibujados.
        """
        output_img = img.copy()
        for (x, y, ancho, alto) in rects:
            cv2.rectangle(output_img, (x, y), (x + ancho, y + alto), (0, 255, 0), 2)
        return output_img

    def draw_img(self, img, axis = "on"):
        """
        Muestra la imagen utilizando matplotlib (conversión de BGR a RGB).
        
        Parámetros:
          img: Imagen a mostrar.
          axis: 'on' o 'off' para mostrar u ocultar los ejes.
        """
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title("Detección de Humanos con HOG")
        if axis == "off":
            plt.axis("off")
        plt.show()

    

    
    def draw_hog_descriptor(self, img, cell_hists, scale_factor=0.5):
        """
        Visualiza el descriptor HOG sobre la imagen dibujando líneas en cada celda.
        
        Parámetros:
        img: Imagen original (preferentemente en color) donde se pintará el descriptor.
        cell_hists: Arreglo 3D de histogramas de celdas obtenido con compute_cell_histograms.
        scale_factor: Factor para escalar la longitud de las líneas dibujadas (ajusta según convenga).
        
        Retorna:
        Una imagen con el overlay del descriptor HOG.
        
        Matemáticamente:
        Para cada celda, se asume que el histograma tiene 'nbins' bins, cada uno representando
        un rango de ángulos de ancho 'bin_width' grados. El ángulo central de cada bin se calcula como:
        
            angle = (bin_index + 0.5) * bin_width
            
        y la longitud de la línea se escala según el valor acumulado en el bin y la magnitud máxima de la celda.
        """
        # Crear una copia para dibujar sobre la imagen
        out_img = img.copy()
        cell_h, cell_w = self.cell_size
        n_cells_y, n_cells_x, _ = cell_hists.shape

        for i in range(n_cells_y):
            for j in range(n_cells_x):
                # Centro de la celda en coordenadas de la imagen
                center_x = int((j + 0.5) * cell_w)
                center_y = int((i + 0.5) * cell_h)
                
                # Obtener el histograma para la celda
                hist = cell_hists[i, j]
                # Normalizar el histograma para que las longitudes sean relativas (evitar división por 0)
                max_val = np.max(hist)
                if max_val == 0:
                    continue
                
                # Dibujar una línea para cada bin
                for bin_idx in range(self.nbins):
                    # Ángulo central del bin (en grados)
                    angle = (bin_idx + 0.5) * self.bin_width
                    angle_rad = np.deg2rad(angle)
                    # Longitud de la línea proporcional al valor del bin (ajustada por scale_factor y el tamaño de la celda)
                    length = (hist[bin_idx] / max_val) * (cell_w / 2) * scale_factor
                    # Calcular las coordenadas de la línea
                    x1 = int(center_x - length * np.cos(angle_rad))
                    y1 = int(center_y - length * np.sin(angle_rad))
                    x2 = int(center_x + length * np.cos(angle_rad))
                    y2 = int(center_y + length * np.sin(angle_rad))
                    cv2.line(out_img, (x1, y1), (x2, y2), (0, 255, 0), 1)
        return out_img

    def draw_hog_descriptor_subset(self, img, cell_hists, max_cells=10, scale_factor=0.7):
        """
        Visualiza el descriptor HOG sobre la imagen dibujando líneas solo en un subconjunto
        de celdas para facilitar la interpretación didáctica.
        
        Parámetros:
        img: Imagen original (preferiblemente en color) donde se pintará el descriptor.
        cell_hists: Arreglo 3D de histogramas de celdas obtenido con compute_cell_histograms.
        max_cells: Número máximo de celdas sobre las que se dibujará el descriptor.
        scale_factor: Factor para escalar la longitud de las líneas dibujadas.
        
        Retorna:
        Imagen con el overlay del descriptor HOG solo en 'max_cells' celdas.
        
        Cada línea dibujada en una celda representa:
        - La orientación del gradiente (dirección de la línea).
        - La magnitud relativa (longitud de la línea) en ese bin del histograma.
        """
        out_img = img.copy()
        cell_h, cell_w = self.cell_size
        n_cells_y, n_cells_x, _ = cell_hists.shape
        
        # Calcula el número total de celdas y selecciona 'max_cells' de ellas de forma uniforme.
        total_cells = n_cells_y * n_cells_x
        indices = np.linspace(0, total_cells - 1, max_cells, dtype=int)
        
        for idx in indices:
            i = idx // n_cells_x
            j = idx % n_cells_x
            
            # Centro de la celda en coordenadas de la imagen.
            center_x = int((j + 0.5) * cell_w)
            center_y = int((i + 0.5) * cell_h)
            hist = cell_hists[i, j]
            max_val = np.max(hist)
            if max_val == 0:
                continue
            
            # Para cada bin, dibuja una línea representando la orientación y magnitud.
            for bin_idx in range(self.nbins):
                # Ángulo central del bin (en grados).
                angle = (bin_idx + 0.5) * self.bin_width
                angle_rad = np.deg2rad(angle)
                # Longitud proporcional a la magnitud del bin (normalizada respecto al valor máximo en la celda).
                length = (hist[bin_idx] / max_val) * (cell_w / 2) * scale_factor
                # Calcular las coordenadas de la línea.
                x1 = int(center_x - length * np.cos(angle_rad))
                y1 = int(center_y - length * np.sin(angle_rad))
                x2 = int(center_x + length * np.cos(angle_rad))
                y2 = int(center_y + length * np.sin(angle_rad))
                cv2.line(out_img, (x1, y1), (x2, y2), (0, 255, 0), 0.5)
        
        return out_img
    
    def draw_cell_grid(self, img, cell_size=(8, 8), color=(255, 0, 0), thickness=1):
        """
        Dibuja líneas de cuadrícula en 'hog_vis' para mostrar las celdas del descriptor HOG.
        
        Parámetros:
        hog_vis: Imagen (con el descriptor HOG dibujado) sobre la que se superpondrán las líneas.
        cell_size: Tupla (cell_w, cell_h) que indica el tamaño de cada celda.
        color: Color de las líneas en formato (B, G, R).
        thickness: Grosor de las líneas.
        
        Retorna:
        out_img: Imagen con la cuadrícula superpuesta.
        """
        out_img = img.copy()
        h, w = out_img.shape[:2]
        cell_w, cell_h = cell_size
        
        # Dibujar líneas horizontales
        for row in range(0, h, cell_h):
            cv2.line(out_img, (0, row), (w, row), color, thickness)
        # Dibujar la última línea horizontal si no coincide exactamente con el borde
        if h % cell_h != 0:
            cv2.line(out_img, (0, h-1), (w, h-1), color, thickness)
        
        # Dibujar líneas verticales
        for col in range(0, w, cell_w):
            cv2.line(out_img, (col, 0), (col, h), color, thickness)
        # Dibujar la última línea vertical si no coincide exactamente con el borde
        if w % cell_w != 0:
            cv2.line(out_img, (w-1, 0), (w-1, h), color, thickness)
        
        return out_img
    



    

