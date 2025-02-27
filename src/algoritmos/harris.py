import cv2
import matplotlib.pyplot as plt
import numpy as np

'''
    Clase que implementa el algoritmo de detecion de esquinas harris.
'''

class harris_detect:

    def __init__(self, gaussbox=3, k=0.04, threshold=0.1, window_size=5):
        '''
        ########################################## PARÁMETROS ##########################################
        
        gaussbox: 
                Type: int
                Description: Describe el tamaño del filtro usado al aplicar el filtro gaussiano. En el valor por defecto el tamaño del filtro seria una matriz 3x3

        k: 
                Type: float
                Description: Variable de control en el calculo del valor harris. Se recomienda valores entre 0.06 y 0.04.DefaultValue=0.04

        threshold: 
                Type: float
                Description: Umbral de los valores harris al aplicar supresion de no maximos. DefaultValue=100

        window_size: 
                Type: float
                Description: Tamaño de la ventana usada en supresion de no maximos. DefaultValue=5
        '''
       
        self.gaussbox = gaussbox
        self.k = k
        self.threshold = 0.1
        self.window_size = window_size 
        
        '''
        ########################################## MÉTODOS ##########################################
        
        entrenar_primera_visita(destino, mapa): 
            Entrena utilizando el método de primera visita. 

        entrenar_cada_visita(destino, mapa): 
            Entrena utilizando el método de cada visita. 

        siguiente_estado(estado, accion): 
            Return: int
            Param: estado -> Estado actual.
                   accion -> Acción tomada.
            Selecciona el siguiente estado en función de la acción y el estado actual usando una política epsilon-greedy.

        generar_episodio(destino, mapa): 
            Return: List
            Param: destino -> Tupla de coordenadas.
                   mapa -> Lista de 0 y 1, donde 0 denota un espacio libre y 1 un obstáculo.
            Genera un episodio (lista de tuplas estado, acción, recompensa).

        es_estado_terminal(estado, destino): 
            Return: bool
            Param: estado -> Tupla de coordenadas.
                   destino -> Tupla de coordenadas.
            Comprueba si el estado es terminal, es decir, si coincide con el destino.

        siguiente_accion(estado): 
            Return: int
            Param: estado -> Estado actual.
            Selecciona la siguiente acción usando una política epsilon-greedy.

        obtener_politica(): 
            Return: List
            Devuelve la política óptima aprendida a partir de los valores Q.
        '''
    
    
    def calc_grad(self, img):
        #Se calcula el gradiente de x e y
        Ix = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        Iy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

        return Ix, Iy

    def calc_grad_prod(self, Ix, Iy):
        #Se calcula los productos de los gradientes
        Ix2 = Ix * Ix
        Iy2 = Iy * Iy
        Ixy = Ix * Iy
        return Ix2, Iy2, Ixy

    def gauss_filter(self, Ix2, Iy2, Ixy, gb):
        #Se aplica un filtro Gaussiano para suavizar
        Ix2 = cv2.GaussianBlur(Ix2, (gb, gb), 1)
        Iy2 = cv2.GaussianBlur(Iy2, (gb, gb), 1)
        Ixy = cv2.GaussianBlur(Ixy, (gb, gb), 1)

    def calc_harris_score(self, Ix2, Iy2, Ixy):
        #Calculamos el Harris Score
        detM = (Ix2 * Iy2) - (Ixy ** 2)
        traceM = Ix2 + Iy2
        R = detM - self.k * (traceM ** 2)

        return R

    def normalize_thresh(self, R):
        #Normalizamos y aplicamos un umbral
        R_norm = cv2.normalize(R, None, 0, 255, cv2.NORM_MINMAX)
        R_norm = np.uint8(R_norm) 
        R_invert = 255 - R_norm
     
        corners = np.where(R_invert >= self.threshold * R_norm.max(), 255, 0)

        return corners

    def non_max_supre(self, R_norm, img):
        R_bin = np.uint8(R_norm)  # Asegúrate de que R_norm esté en el rango correcto 0-255

        # Llamamos a connectedComponentsWithStats con la imagen binaria
        r, l, s, centroids = cv2.connectedComponentsWithStats(R_bin)  # Aquí es importante usar la imagen binaria

        # Definir criterios para cornerSubPix
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)

        # Encontrar las esquinas más fuertes con cornerSubPix
        strongest_corners = cv2.cornerSubPix(img, np.float32(centroids), (5,5), (-1, -1), criteria)
        
        return np.uint8(strongest_corners)
    
    
    def draw_corner(self, strongest_corners, img):
        # Dibujar las esquinas detectadas en la imagen original
        output_image = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # Crear una copia para no modificar la imagen original 
        A = np.array([[0,100], [100, 100], [0,200]])

        #Iterar sobre las tuplas (x, y) de strongest_corners
        for y, x in np.argwhere(strongest_corners == 255) :  # strongest_corners es una lista de tuplas (x, y)
            cv2.circle(output_image, (x, y), 1, (0, 0, 255), -1)  # Dibujar un círculo rojo en la esquina

        return output_image
    
    def draw_corner_p(self, strongest_corners, img):
        # Dibujar las esquinas detectadas en la imagen original
        output_image = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # Crear una copia para no modificar la imagen original 
        A = np.array([[0,100], [100, 100], [0,200]])

        #Iterar sobre las tuplas (x, y) de strongest_corners
        for x, y in strongest_corners :  # strongest_corners es una lista de tuplas (x, y)
            cv2.circle(output_image, (x, y), 1, (0, 0, 255), -1)  # Dibujar un círculo rojo en la esquina

        return output_image
 

       
    def draw_img(self, img, axis="on"): 
        # Dibuja la imagen con o sin ejes
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title("Puntos detectados por el Algoritmo")

        if(axis == "off"): 
            plt.axis("off")

        plt.show()

    def draw_values_harris(strongest_corners, axis="on"):   
        #Muestra la variacion del Harris Score en cada pixel
        plt.imshow(strongest_corners, cmap='jet') 
        plt.colorbar(label="Harris Score")
        
        if(axis == "off"): 
                plt.axis("off")

        plt.title("Mapa de Harris Score")
        plt.show()

'''
    #Harris score propio de openCV
    img2 = cv2.cornerHarris(img,2,3,0.04)
    plt.imshow(img2, cmap='jet') 
    plt.colorbar(label="Harris Score")
    plt.axis("off")
    plt.title("Mapa de Harris Score")
    plt.show()
'''
