"""
Demo de detección de EPP con modelo YOLOv11 sin threading
"""
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

# Etiquetas del modelo
LABELS = [
    "Person", "goggles", "helmet",
    "no-goggles", "no-helmet",
    "no-vest", "vest"
]


def load_model(model_path: Path) -> YOLO:
    """Carga el modelo YOLO desde disco."""
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo no encontrado en: {model_path}")
    return YOLO(str(model_path))


def draw_detections(frame, detections) -> None:
    """Dibuja cajas y etiquetas en el frame."""
    for box in detections.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        label = LABELS[cls]
        # Verde si cumple, rojo si es 'no-...'
        color = (0, 255, 0) if not label.startswith("no-") else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"{label} {conf:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1
        )


def main():
    # Fuente de video: argumento o archivo por defecto
    video_src = sys.argv[1] if len(sys.argv) > 1 else "epp.mp4"
    cap = cv2.VideoCapture(video_src)
    if not cap.isOpened():
        print(f"Error: no se pudo abrir el video o dispositivo {video_src}")
        return

    # Carga del modelo
    model_path = Path(__file__).parent / "model" / "best.pt"
    model = load_model(model_path)

    print("Iniciando demo. Pulsa 'q' para salir.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Fin del video o error al leer el frame.")
            break

        # Reducir resolución para acelerar (opcional)
        frame_small = cv2.resize(frame, (0, 0), fx=0.7, fy=0.7)

        # Inferencia
        results = model(frame_small)[0]

        # Dibujar detecciones
        draw_detections(frame_small, results)

        # Mostrar
        cv2.imshow("Detección EPP (YOLOv11)", frame_small)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Demo interrumpido por el usuario.")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
