import cv2
import matplotlib.pyplot as plt
import numpy as np

'''
    Clase que implementa el algoritmo de detecion de esquinas harris.
'''

class harris_detect:

    def __init__(self, gaussbox=3, k=0.04, threshold=0.1, window_size=5, window_size_mx=7):
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
                Description: Umbral de los valores harris al normalizar. DefaultValue=0.1%

        window_size: 
                Type: int
                Description: Tamaño de la ventana usada en supresion de no maximos. DefaultValue=7
        '''
       
        self.gaussbox = gaussbox
        self.k = k
        self.threshold = 0.1
        self.window_size = window_size
        
        '''
        ########################################## MÉTODOS ##########################################

        calc_grad(img): 
            Return: List
            Param: img -> Matriz de la imagen.
            Calcula el gradiente en las direcciones x e y de la imagen.

        calc_grad_prod(Ix, Iy):
            Return: List
            Param: 
                Ix -> Gradiente en la dirección x.
                Iy -> Gradiente en la dirección y.
            Calcula el producto de los gradientes. Devuelve tres matrices: Ix², Iy² e Ixy.

        gauss_filter(Ix2, Iy2, Ixy, gb):
            Return: List
            Param: 
                Ix2 -> Cuadrado del gradiente en x.
                Iy2 -> Cuadrado del gradiente en y.
                Ixy -> Producto de ambos gradientes.
                gb -> Kernel del filtro gaussiano.
            Aplica un filtro gaussiano para suavizar la imagen.

        calc_harris_score(Ix2, Iy2, Ixy): 
            Return: List
            Param: 
                Ix2 -> Cuadrado del gradiente en x.
                Iy2 -> Cuadrado del gradiente en y.
                Ixy -> Producto de ambos gradientes.
            Calcula la puntuación de Harris para cada píxel (x, y).

        normalize_thresh(self, R):
            Return: List
            Param: 
                R -> Matriz de las puntuaciones de Harris.
            Normaliza la matriz R para que sus valores estén en el rango [0,255] y aplica un umbral para descartar valores no válidos.

        non_max_supre(R_norm, window_size):
            Return: List
            Param: 
                R_norm -> Matriz normalizada después de aplicar un umbral a las puntuaciones de Harris.
                window_size -> Tamaño de la ventana utilizada en la supresión de no máximos.
            Aplica la supresión de no máximos iterando sobre cada píxel y conservando solo el de mayor puntuación dentro de cada ventana.

        draw_corner(strongest_corners, img): 
            Return: List
            Param: 
                strongest_corners -> Matriz con las puntuaciones de Harris destacadas.
                img -> Matriz de la imagen.   
            Dibuja círculos rojos en las posiciones de las esquinas detectadas.

        draw_img(img, axis="on"): 
            Param: 
                img -> Matriz de la imagen.
                axis -> Controla si se debe mostrar el eje de coordenadas en la imagen.
            Muestra la imagen en pantalla.

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

        return Ix2, Iy2, Ixy

    def calc_harris_score(self, Ix2, Iy2, Ixy):
        
        #Calculamos el Harris Score
        detM = (Ix2 * Iy2) - (Ixy ** 2)
        traceM = Ix2 + Iy2
        R = detM - self.k * (traceM ** 2)

        return R

    def normalize_thresh(self, R):
        
        # Aplica un umbral
        R[R < self.threshold * np.max(R)] = 0  

        # Normalizar al rango [0, 255]
        R_norm = cv2.normalize(R, None, 0, 255, norm_type=cv2.NORM_MINMAX)
        
        
        return R_norm

    def non_max_supre(self, R_norm, window_size):
        strongest_corners = np.zeros(R_norm.shape)  # Matriz vacía para almacenar los máximos locales
        offset = window_size // 2  # Offset para centrar la ventana
        
        # Iterar sobre cada píxel de la imagen
        for x, y in np.argwhere(R_norm > 0):  # Solo recorrer donde hay respuestas (>0)
            best_point = 0
            bx, by = x, y  # Inicializar los valores para almacenar el mejor punto

            # Iterar sobre la ventana alrededor de (x,y)
            for i in range(-offset, offset + 1):
                for j in range(-offset, offset + 1):
                    nx, ny = x + i, y + j  # Coordenadas dentro de la ventana
                    
                    # Verificar si la posición está dentro de la imagen
                    if 0 <= nx < R_norm.shape[0] and 0 <= ny < R_norm.shape[1]:  
                        if R_norm[nx, ny] > best_point:
                            best_point = R_norm[nx, ny]
                            bx, by = nx, ny

            # Si encontramos un máximo en la ventana, lo guardamos en la nueva imagen
            if best_point > 0:
                strongest_corners[bx, by] = best_point

        return np.uint8(strongest_corners)


    def draw_corner(self, strongest_corners, img):
        
        # Dibujar las esquinas detectadas en la imagen original
        output_image = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # Crear una copia para no modificar la imagen original 
        
        #Iterar sobre las tuplas (x, y) de strongest_corners
        for (y, x) in np.argwhere(strongest_corners != 0):  # strongest_corners es una lista de tuplas (x, y)
            cv2.circle(output_image, (x, y), 1, (0, 0, 255), -1)  # Dibujar un círculo rojo en la esquina

        return output_image
    
    
    def draw_img(self, img, axis="on"): 
        
        # Dibuja la imagen con o sin ejes
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title("Puntos detectados por el Algoritmo")

        if(axis == "off"): 
            plt.axis("off")

        plt.show()   

