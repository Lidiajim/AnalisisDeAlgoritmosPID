# 📷 Comparativa de Técnicas de Procesamiento de Imágenes para la Detección de Peatones con SVM 

## 📝 Descripción del Proyecto  
Este proyecto tiene como objetivo entrenar un modelo de **aprendizaje automático SVM (Support Vector Machine)** utilizando un banco de imágenes con y sin personas. Posteriormente, se aplicarán cuatro algoritmos de extracción de características:  
- **Harris Corner Detection**  
- **SURF (Speeded-Up Robust Features)**  
- **SIFT (Scale-Invariant Feature Transform)**  
- **HOG (Histogram of Oriented Gradients)**  

El modelo entrenado será evaluado en un conjunto de imágenes por cada algoritmo, para determinar si es capaz de predecir la presencia de personas y se calculará un **porcentaje de eficiencia** para cada método.

Además, el proyecto incluye notebooks didácticos que explican paso a paso el funcionamiento de cada algoritmo, junto con su implementación detallada

## 🚀 Uso
Para ejecutar los notebooks del proyecto, asegúrese de que tiene instalado Python 3.9 o superior y las dependencias necesarias.  

### Ejecutar un notebook  

1. Asegúrese de estar en la raíz del proyecto.  
2. Active el entorno virtual si lo ha creado (ver sección de instalación).
## 🛠️ Instalación  
Para ejecutar este proyecto, es necesario instalar las dependencias especificadas en `requirements.txt`. Se recomienda usar un entorno virtual. 

```bash
# Crear el entorno virtual
python -m venv env

# Activar el entorno virtual
# En Windows:
env\Scripts\activate

# En macOS/Linux:
source env/bin/activate

# Instalar dependencias
pip install -r requirements.txt