import os
import xml.etree.ElementTree as ET

annotations_path = "C:/Users/vflor/OneDrive/ING.Soft/4año/PID/IA/test"  # Ruta de tus anotaciones
classes = set()

for xml_file in os.listdir(annotations_path):
    if xml_file.endswith(".xml"):  # Asegurar que solo leemos archivos XML
        xml_path = os.path.join(annotations_path, xml_file)
        
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        for obj in root.findall("object"):
            class_name = obj.find("name").text
            classes.add(class_name)

# Mostrar las clases encontradas
print("Clases detectadas en el dataset:", classes)
