import cv2
import matplotlib.pyplot as plt

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
