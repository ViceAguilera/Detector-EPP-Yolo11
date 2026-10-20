# Detector de EPP en tiempo real

Aplicación de escritorio que detecta elementos de protección personal (EPP) en video usando YOLOv11, con seguimiento de personas mediante ByteTrack.

![Python](https://img.shields.io/badge/Python-3.9-3776AB?style=flat&logo=python&logoColor=white)
![YOLOv11](https://img.shields.io/badge/YOLO-v11-00FFFF?style=flat)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=opencv&logoColor=white)
![Estado](https://img.shields.io/badge/estado-activo-brightgreen?style=flat)
![Licencia](https://img.shields.io/badge/licencia-MIT-blue?style=flat)
![Último commit](https://img.shields.io/github/last-commit/ViceAguilera/Detector-EPP-Yolo11?style=flat)

<!-- Demo: agrega una captura o GIF en docs/demo.gif y descomenta la línea siguiente -->
<!-- <p align="center"><img src="docs/demo.gif" alt="Demostración del detector de EPP" width="800"></p> -->

## Tabla de contenidos

- [Descripción](#descripción)
- [Estado del proyecto](#estado-del-proyecto)
- [Características](#características)
- [Tecnologías](#tecnologías)
- [Requisitos](#requisitos)
- [Instalación](#instalación)
- [Uso](#uso)
- [Estructura del proyecto](#estructura-del-proyecto)
- [Licencia](#licencia)
- [Autor](#autor)

## Descripción

El programa procesa un video o un dispositivo de captura cuadro a cuadro, detecta personas y su equipamiento de seguridad con un modelo YOLOv11 entrenado a medida (`model/best.pt`) y asigna un identificador persistente a cada detección con ByteTrack. Una interfaz gráfica muestra el video anotado y un resumen por persona con los EPP detectados o ausentes. El proyecto se encuentra en desarrollo (versión beta).

## Estado del proyecto

**Activo.** El proyecto está en desarrollo (versión beta) y recibe cambios.

## Características

- Detección con YOLOv11 (umbral de confianza 0.5).
- Seguimiento multi-objeto con ByteTrack, con ID visible sobre cada detección.
- Resumen por persona de los pares de EPP: `goggles`/`no-goggles`, `helmet`/`no-helmet` y `vest`/`no-vest`.
- Asociación de cada EPP a la persona más cercana (umbral de 300 píxeles).
- Interfaz Tkinter con panel de resumen y botón de pausa/reanudación.
- Acepta un archivo de video o un dispositivo de captura como fuente.

## Tecnologías

- Python 3.9
- [Ultralytics](https://ultralytics.com/) (YOLOv11)
- [OpenCV](https://opencv.org/)
- [ByteTrack](https://github.com/cj-mills/cjm-byte-track) (`cjm_byte_track`)
- Tkinter y Pillow para la interfaz

## Requisitos

- Python 3.9
- Dependencias de `requirements.txt` (`ultralytics`, `opencv-python`, `bytetracker`).
- El código importa `cjm_byte_track`; si falta, el programa indica instalarlo con `pip install cjm_byte_track`.
- Opcional: GPU NVIDIA con CUDA 11.8 para acelerar la inferencia.

## Instalación

```bash
git clone https://github.com/ViceAguilera/Detector-EPP-Yolo11.git
cd Detector-EPP-Yolo11
python -m venv venv
source venv/bin/activate        # En Windows: ./venv/Scripts/activate
pip install -r requirements.txt
```

En Linux, OpenCV puede requerir `sudo apt-get install -y libgl1-mesa-glx`.

Para usar PyTorch con CUDA 11.8:

```bash
pip uninstall -y torch torchvision torchaudio
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Uso

```bash
# Con el video de ejemplo incluido (epp.mp4)
python main.py

# Con otro video o dispositivo de captura
python main.py ruta/al/video.mp4
python main.py 0
```

## Estructura del proyecto

```
main.py            # Punto de entrada: captura, inferencia y actualización de la interfaz
tracker.py         # Carga del modelo, ByteTrack, dibujo y resumen de EPP por persona
ui.py              # Interfaz gráfica en Tkinter
model/best.pt      # Pesos del modelo YOLOv11 entrenado
epp.mp4            # Video de ejemplo
requirements.txt   # Dependencias
```

## Licencia

Distribuido bajo la licencia MIT. Ver [LICENSE](LICENSE).

## Autor

**Vicente Aguilera Arias** — [Portafolio](https://portfolio.iamsirius.cl) · [GitHub](https://github.com/ViceAguilera) · [LinkedIn](https://www.linkedin.com/in/vicenteaguilera/)
