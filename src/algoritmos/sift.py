import cv2
import matplotlib.pyplot as plt
import numpy as np

'''
    Clase que implementa el algoritmo SIFT (Scale-Invariant Feature Transform)
    para la detección y descripción de características en imágenes.
'''

class sift_detect:
    
    def __init__(self, nfeatures=0, nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6):
        '''
        ########################################## PARÁMETROS ##########################################
        
        nfeatures:
            Type: int
            Description: Número máximo de características a detectar. Si es 0, se detectan todas las posibles.
        
        nOctaveLayers:
            Type: int
            Description: Número de capas en cada octava de la pirámide de escalas.
        
        contrastThreshold:
            Type: float
            Description: Umbral para descartar características de bajo contraste.
        
        edgeThreshold:
            Type: float
            Description: Umbral para descartar características en bordes.
        
        sigma:
            Type: float
            Description: Desviación estándar inicial para el filtro Gaussiano aplicado.
        '''
        self.nfeatures = nfeatures
        self.nOctaveLayers = nOctaveLayers
        self.contrastThreshold = contrastThreshold
        self.edgeThreshold = edgeThreshold
        self.sigma = sigma
        
        # Crear el objeto SIFT con los parámetros especificados.
        self.sift = cv2.SIFT_create(nfeatures=self.nfeatures, 
                                    nOctaveLayers=self.nOctaveLayers, 
                                    contrastThreshold=self.contrastThreshold, 
                                    edgeThreshold=self.edgeThreshold, 
                                    sigma=self.sigma)
    
    def detect_and_compute(self, img):
        '''
        detect_and_compute(img):
            Return: keypoints, descriptors
            Param: 
                img -> Matriz de la imagen. Si es a color, se convertirá a escala de grises.
            Detecta los puntos clave y calcula sus descriptores utilizando SIFT.
        '''
        # Verificar si la imagen es a color y convertirla a escala de grises
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        # Detectar los puntos clave y calcular los descriptores
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        
        return keypoints, descriptors

    def draw_keypoints(self, img, keypoints, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS):
        '''
        draw_keypoints(img, keypoints, flags):
            Return: Imagen con los keypoints dibujados.
            Param:
                img -> Matriz de la imagen original.
                keypoints -> Lista de puntos clave detectados.
                flags -> Opciones para dibujar (por defecto se dibujan el tamaño y la orientación).
            Dibuja los puntos clave sobre la imagen y devuelve la imagen resultante.
        '''
        output_img = cv2.drawKeypoints(img, keypoints, None, flags=flags)
        
       
        return output_img
    
    def show_img_keypoints(self, draw_key):
         # Mostrar la imagen con los keypoints detectados
        plt.figure(figsize=(10,8))
        plt.imshow(cv2.cvtColor(draw_key, cv2.COLOR_BGR2RGB))
        plt.title("Puntos Clave Detectados con SIFT")
        plt.axis("off")
        plt.show()
    
    def show_image(self, img, title="Imagen con Keypoints", axis="on"):
        '''
        show_image(img, title, axis):
            Param:
                img -> Matriz de la imagen a mostrar.
                title -> Título de la ventana de la imagen.
                axis -> Controla la visualización de los ejes ("on" o "off").
            Muestra la imagen utilizando matplotlib.
        '''
        # Si la imagen es BGR (color), convertir a RGB para visualizarla correctamente
        if len(img.shape) == 3:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = img
        
        plt.imshow(img_rgb, cmap="gray")
        plt.title(title)
        if axis == "off":
            plt.axis("off")
        plt.show()
    
    def show_dog_pyramid(self, dog_pyr):
        """
        Muestra la pirámide de Diferencia de Gaussianas (DoG) en una sola figura, 
        organizando subplots en forma de rejilla:
        - Filas = número de octavas
        - Columnas = número de niveles por octava
        """
        for i, octave in enumerate(dog_pyr):
            print(f"Octava {i}:")
            for j, level in enumerate(octave):
                print(f"  Nivel {j}: resolución {level.shape}")
        num_octaves = len(dog_pyr)
        # Se asume que cada octava tiene la misma cantidad de niveles en la DoG
        num_levels = len(dog_pyr[0])
        
        fig, axes = plt.subplots(nrows=num_octaves, 
                                 ncols=num_levels, 
                                 figsize=(3 * num_levels, 3 * num_octaves))
        
        for i in range(num_octaves):
            for j in range(num_levels):
                img = dog_pyr[i][j]
                axes[i, j].imshow(img, 
                                  cmap="gray", 
                                  interpolation='nearest',
                                  extent=[0, img.shape[1], img.shape[0], 0])
                axes[i, j].set_aspect('equal')
                axes[i, j].axis("off")
                
                # Título de columna (solo en la primera fila)
                if i == 0:
                    axes[i, j].set_title(f"Nivel {j}")
            
            # Etiqueta de fila (solo en la primera columna)
            axes[i, 0].set_ylabel(f"Octava {i}", rotation=90, size="large")
        
        plt.tight_layout()
        plt.show()
        
    def build_dog_pyramid(self, gaussian_pyramid):
        """
        Construye la pirámide de Diferencia de Gaussiana (DoG) a partir de la pirámide Gaussiana.
        Por cada octava, se calcula la diferencia entre imágenes Gaussianas consecutivas.
        """
        dog_pyr = []
        for octave_images in gaussian_pyramid:
            dog_images = []
            for i in range(len(octave_images) - 1):
                dog = octave_images[i + 1] - octave_images[i]
                dog_images.append(dog)
            dog_pyr.append(dog_images)
        return dog_pyr
    
    def show_gaussian_pyramid(self, gaussian_pyr):
        """
        Muestra la pirámide Gaussiana en una sola figura, 
        organizando subplots en forma de rejilla:
        - Filas = número de octavas
        - Columnas = número de niveles por octava

        Ajusta la interpolación a 'nearest' y el aspect a 'equal'
        para visualizar mejor la diferencia de resolución.
        """
        for i, octave in enumerate(gaussian_pyr):
            print(f"Octava {i}:")
            for j, level in enumerate(octave):
                print(f"  Nivel {j}: resolución {level.shape}")
        num_octaves = len(gaussian_pyr)
        # Suponiendo que todas las octavas tienen el mismo número de niveles
        num_levels = len(gaussian_pyr[0])
    
        fig, axes = plt.subplots(nrows=num_octaves, 
                                 ncols=num_levels, 
                                 figsize=(3 * num_levels, 3 * num_octaves))
    
        for i in range(num_octaves):
            for j in range(num_levels):
                img = gaussian_pyr[i][j]
                axes[i, j].imshow(img, 
                                  cmap="gray", 
                                  interpolation='nearest',
                                  extent=[0, img.shape[1], img.shape[0], 0])
                axes[i, j].set_aspect('equal')
                axes[i, j].axis("off")
    
                if i == 0:
                    axes[i, j].set_title(f"Nivel {j}")
    
            axes[i, 0].set_ylabel(f"Octava {i}", rotation=90, size="large")
    
        plt.tight_layout()
        plt.show()
        
    def build_gaussian_pyramid(self, base_img, num_octaves=4, num_scales=3, sigma=1.6):
        """
        Construye la pirámide Gaussiana.
        - base_img: imagen base (float32) en escala de grises.
        - num_octaves: número de octavas.
        - num_scales: número de niveles por octava (sin contar los extra para DoG).
        - sigma: sigma inicial para la primera escala de la primera octava.
        
        Devuelve una lista (de tamaño num_octaves), donde cada elemento es otra lista 
        con (num_scales + 3) imágenes Gaussianas (por cada octava).
        """
        k = 2 ** (1.0 / num_scales)  # Factor de escala entre niveles
        pyr = []
        for octave in range(num_octaves):
            gaussian_images = []
            
            if octave == 0:
                current_img = base_img
            else:
                current_img = cv2.pyrDown(pyr[octave - 1][num_scales])
            
            for scale in range(num_scales + 3):
                sigma_curr = sigma * (k ** scale)
                gaussian_blurred = cv2.GaussianBlur(current_img, (0, 0), sigma_curr)
                gaussian_images.append(gaussian_blurred)
            
            pyr.append(gaussian_images)
        return pyr

    def to_grayscale_float32(self, img):
        """
        Convierte la imagen a escala de grises en formato float32 [0,1].
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return np.float32(gray) / 255.0

    def compute_gradient(self, patch):
        """
        Calcula la magnitud y el ángulo del gradiente en una ventana (patch).
        Retorna:
          - magnitude: matriz con la magnitud del gradiente.
          - angle: matriz con el ángulo en grados, en el rango [0, 360).
        """
        gx = cv2.Sobel(patch, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(patch, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(gx**2 + gy**2)
        angle = np.arctan2(gy, gx) * 180 / np.pi  # en grados
        angle[angle < 0] += 360  # Asegurar valores en [0,360)
        return magnitude, angle

    def compute_orientation_histogram(self, magnitude, angle, num_bins=36):
        """
        Construye un histograma de orientaciones a partir de la magnitud y el ángulo.
        Cada píxel contribuye al bin correspondiente según su ángulo ponderado por su magnitud.
        """
        histogram = np.zeros(num_bins)
        bin_width = 360.0 / num_bins
        for i in range(magnitude.shape[0]):
            for j in range(magnitude.shape[1]):
                bin_idx = int(np.floor(angle[i, j] / bin_width)) % num_bins
                histogram[bin_idx] += magnitude[i, j]
        return histogram

    def dominant_orientation(self, histogram, num_bins=36):
        """
        Determina la orientación dominante a partir del histograma.
        Retorna el ángulo correspondiente al bin con mayor valor.
        """
        bin_width = 360.0 / num_bins
        max_bin = np.argmax(histogram)
        return max_bin * bin_width

    def demonstrate_orientation(self, patch):
        """
        Demuestra de forma didáctica el proceso de cálculo de la orientación dominante.
        Muestra:
          1. La ventana del Keypoint.
          2. El histograma de orientaciones.
          3. La ventana con una flecha que indica la orientación (invirtiendo el ángulo).
        Se asume que 'patch' es una imagen en escala de grises.
        """
        magnitude, angle = self.compute_gradient(patch)
        histogram = self.compute_orientation_histogram(magnitude, angle, num_bins=36)
        dom_angle = self.dominant_orientation(histogram, num_bins=36)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # 1. Mostrar la ventana original
        axes[0].imshow(patch, cmap='gray')
        axes[0].set_title("Ventana del Keypoint")
        axes[0].axis('off')
        
        # 2. Mostrar el histograma de orientaciones
        axes[1].bar(np.arange(36) * (360/36), histogram, width=(360/36))
        axes[1].set_title("Histograma de Orientaciones")
        axes[1].set_xlabel("Ángulo (°)")
        axes[1].set_ylabel("Suma de magnitudes")
        axes[1].axvline(dom_angle, color='r', linestyle='--', label=f"Dominante: {dom_angle:.1f}°")
        axes[1].legend()
        
        # 3. Mostrar la ventana con la flecha de orientación (invirtiendo el ángulo)
        axes[2].imshow(patch, cmap='gray')
        center = (patch.shape[1] // 2, patch.shape[0] // 2)
        arrow_length = 20
        
        # Invertir el ángulo
        draw_angle = 360 - dom_angle
        theta = np.deg2rad(draw_angle)
        
        x2 = center[0] + int(arrow_length * np.cos(theta))
        y2 = center[1] - int(arrow_length * np.sin(theta))
        
        axes[2].arrow(center[0], center[1], x2 - center[0], y2 - center[1],
                      color='red', width=1.5, head_width=5)
        axes[2].set_title(f"Orientación Dominante (Invertida): {dom_angle:.1f}°")
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()
