# Detector de EPP en tiempo real 

###### **🚧 V1.0 Beta development.🚧🔨**

_Sistema de reconocimiento automático de matrículas vehiculares y API para el control de acceso en tiempo real_
_Sistema de detección de EPP en tiempo real utilizando YOLOv11 y OpenCV._

## Construido con 🛠️

- [Python v3.9](https://www.python.org/) - Lenguaje de programación
- [OpenCV](https://opencv.org/) - Librería de visión artificial
- [Ultralytics](https://ultralytics.com/) - Librería de modelo de detección de objetos YOLOv11

## Comenzando 🚀

### Instalacion  🔧

<details>
   <summary>Linux</summary>

1. Se debe instalar venv
    ```bash
    sudo apt-get install python3.9-venv
    ```
  
2. Se debe instalar un packete para OpenCV
    ```bash
    sudo apt-get update && apt-get install -y libgl1-mesa-glx
    ```

3. Se clona el repositorio de GitHub
    ```bash
    git clone https://github.com/ViceAguilera/Detector-EPP-Yolo11.git
    ```
  
4. Se ingresa a la carpeta del proyecto
    ```bash
    cd detector-epp
    ```
  
5. Se crea un entorno virtual
    ```bash
    python3.9 -m venv venv
    ```
    
6. Se activa el entorno virtual
    ```bash
    source venv/bin/activate
    ```

7. Se instala los requerimientos del proyecto
    ```bash
    pip install -r requirements.txt
    ```
   
8. Se desinstala pytorch
    ```bash
    pip uninstall -y torch torchvision torchaudio
    ```

8. Se instala CUDA Pytorch
    ```bash
   pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

9. Se ejecuta el script
    ```bash
    python3.9 main.py
    ```
</details>

<details>
  <summary>Windows</summary>

1. Se clona el repositorio de GitHub
    ```bash
    git clone https://github.com/ViceAguilera/Detector-EPP-Yolo11.git detector-epp
    ```
  
2. Se ingresa a la carpeta del proyecto
    ```bash
    cd detector-epp
    ```
  
3. Se crea un entorno virtual
    ```bash
    python -m venv venv
    ```
    
4. Se activa el entorno virtual
    ```bash
    ./venv/Scripts/activate
    ```
   
5. Se instala los requerimientos del proyecto
    ```bash
    pip install -r requirements.txt
    ```
6. Se desinstala pytorch
    ```bash
    pip uninstall -y torch torchvision torchaudio
    ```

7. Se instala CUDA Pytorch
    ```bash
   pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

8. Se ejecuta el script
    ```bash
    python3.9 main.py
    ```
</details>
   
## Licencia 📄

Este proyecto está bajo el _GNU AFFERO GENERAL PUBLIC LICENSE_ - mira el archivo [LICENSE](LICENSE) para detalles

## Autor ✒️
[**Vicente Aguilera Arias**](https://github.com/ViceAguilera)