import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('img/peatones4.jpg',cv2.IMREAD_GRAYSCALE)

#Se calcula el gradiente de x e y
Ix = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
Iy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

#Se calcula los productos de los gradientes
Ix2 = Ix * Ix
Iy2 = Iy * Iy
Ixy = Ix * Iy

#Se aplica un filtro Gaussiano para suavizar
Ix2 = cv2.GaussianBlur(Ix2, (3, 3), 1)
Iy2 = cv2.GaussianBlur(Iy2, (3, 3), 1)
Ixy = cv2.GaussianBlur(Ixy, (3, 3), 1)

#Calculamos el Harris Score
k = 0.04
detM = (Ix2 * Iy2) - (Ixy ** 2)
traceM = Ix2 + Iy2
R = detM - k * (traceM ** 2)

#Normalizamos y aplicamos un umbral
R_norm = cv2.normalize(R, None, 0, 255, cv2.NORM_MINMAX)
R_norm = np.uint8(R_norm)
threshold = 100
corners = R_norm > threshold

#Aplicamos supresion de no Maximos
window_size = 5
dilated = cv2.dilate(R, np.ones((window_size, window_size), np.uint8))  # Máximo local
strongest_corners = (R == dilated) & corners  # Solo conservar los valores máximos

# Dibujar las esquinas detectadas después de la NMS
output_image = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
for y, x in np.argwhere(strongest_corners):  
    cv2.circle(output_image, (x, y), 2, (0, 0, 255), -1)  # Dibujar círculo en la esquina

# Mostrar imagen con esquinas detectadas
plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()

#Muestra la variacion del Harris Score en cada pixel
plt.imshow(R, cmap='jet') 
plt.colorbar(label="Harris Score")
plt.axis("off")
plt.title("Mapa de Harris Score")
plt.show()


#Harris score propio de openCV
img2 = cv2.cornerHarris(img,2,3,0.04)
plt.imshow(img2, cmap='jet') 
plt.colorbar(label="Harris Score")
plt.axis("off")
plt.title("Mapa de Harris Score")
plt.show()

