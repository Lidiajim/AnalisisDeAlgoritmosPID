import cv2
import matplotlib.pyplot as plt
import numpy as np

class hog_detect:
    def __init__(self, cell_size=(8,8), block_size=(2,2), block_stride=(8,8), nbins=9,
                 win_stride=(8,8), padding=(16,16), scale=1.05, bin_width=20):
        # Inicialización de parámetros
        self.cell_size = cell_size
        self.block_size = block_size
        self.block_stride = block_stride
        self.nbins = nbins
        self.win_stride = win_stride
        self.padding = padding
        self.scale = scale
        self.bin_width = bin_width
        
        # Tamaño de la ventana de detección (64x128 para peatones)
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
        if len(img.shape) > 2:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        return gray

    def compute_gradients(self, img):
        Gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
        Gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = np.sqrt(Gx**2 + Gy**2)
        orientation = np.arctan2(Gy, Gx) * (180 / np.pi)
        orientation[orientation < 0] += 180
        return magnitude, orientation

    def cell_histogram(self, cell_mag, cell_ori):
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
        block_h, block_w = self.block_size
        n_cells_y, n_cells_x, _ = cell_hists.shape
        n_blocks_y = n_cells_y - block_h + 1
        n_blocks_x = n_cells_x - block_w + 1
        hog_vector = []
        for i in range(n_blocks_y):
            for j in range(n_blocks_x):
                block = cell_hists[i:i+block_h, j:j+block_w, :].flatten()
                norm = np.linalg.norm(block) + 1e-6
                block_norm = block / norm
                hog_vector.extend(block_norm)
        return np.array(hog_vector)

    def compute_hog(self, img):
        gray = self.to_grayscale(img)
        magnitude, orientation = self.compute_gradients(gray)
        cell_hists = self.compute_cell_histograms(magnitude, orientation)
        hog_descriptor = self.normalize_blocks(cell_hists)
        return hog_descriptor

    def detect_humans(self, img, threshold=0.5):
        """
        Detección estándar de humanos usando el HOG de OpenCV.
        """
        rects, weights = self.hog.detectMultiScale(
            img,
            winStride=self.win_stride,
            padding=self.padding,
            scale=self.scale
        )
        
        # Filtrar detecciones por un umbral
        filtered_rects = []
        filtered_weights = []
        for (rect, w) in zip(rects, weights):
            if w > threshold:
                filtered_rects.append(rect)
                filtered_weights.append(w)
        return filtered_rects, filtered_weights

    def detect_humans_individual(self, img, threshold=0.5, sub_threshold=0.5):
        """
        Versión 'forzada' para intentar separar peatones muy juntos.
        
        1) Detecta peatones de la forma habitual.
        2) Si un rectángulo es muy ancho, recorta esa región y vuelve a aplicar
           la detección con parámetros más finos (por ejemplo, menor stride o scale).
        3) Devuelve las detecciones individuales en lugar de un rectángulo grande.
        
        Parámetros:
          threshold -> Umbral de confianza para la detección principal.
          sub_threshold -> Umbral de confianza para la detección en sub-ROI.
        
        Nota: Este método es un truco y no garantiza que se separen
              correctamente peatones muy pegados si el modelo
              en sí no los distingue.
        """
        # 1) Detección principal
        rects, weights = self.detect_humans(img, threshold=threshold)
        
        final_rects = []
        final_weights = []

        # 2) Recorremos cada rectángulo
        for (rect, w) in zip(rects, weights):
            x, y, w_rect, h_rect = rect

            # Heurística: si el ancho es muy grande, asumimos que podría haber varias personas
            # Ajusta el "umbral de ancho" a tu criterio
            # Por ejemplo, si > 80 px, volvemos a hacer subdetección
            if w_rect > 60:
                # Extraer sub-ROI
                subROI = img[y:y+h_rect, x:x+w_rect]
                
                # Realizamos una segunda detección con parámetros más finos
                # por ejemplo, stride=(2,2), scale=1.01
                sub_rects, sub_weights = self.hog.detectMultiScale(
                    subROI,
                    winStride=(2,2),
                    padding=self.padding,
                    scale=1.005
                )
                
                # Filtrar detecciones en subROI
                for (srect, sw) in zip(sub_rects, sub_weights):
                    if sw > sub_threshold:
                        sx, sy, sw_rect, sh_rect = srect
                        # Ajustar coords al original
                        final_rects.append((x+sx, y+sy, sw_rect, sh_rect))
                        final_weights.append(sw)
                
                # Si la subdetección no encontró nada, mantenemos el rectángulo original
                if sub_rects is None or sub_rects.size == 0:
                    final_rects.append(rect)
                    final_weights.append(w)
            else:
                # Si el rect no es "muy ancho", lo conservamos
                final_rects.append(rect)
                final_weights.append(w)
        
        return final_rects, final_weights

    def draw_detections(self, img, rects):
        output_img = img.copy()
        for (x, y, w, h) in rects:
            cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return output_img

    def draw_img(self, img, axis="on"):
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title("Detección de Humanos con HOG")
        if axis == "off":
            plt.axis("off")
        plt.show()
