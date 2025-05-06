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
    """
    Dibuja cajas y etiquetas en el frame, pero para cada par
    (sí/no) de EPP sólo se muestra uno:
      - Si 'no-xxx' existe con conf >= 0.5 → dibuja sólo 'no-xxx'
      - Else, si 'xxx' existe con conf >= 0.5 → dibuja 'xxx'
    Además siempre se muestran los 'Person'.
    """
    # 1. Recolectar todas las detecciones > 0.5 en un dict por etiqueta
    det_map = {}  # label -> list of dicts {bbox, conf}
    for box in detections.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        if conf < 0.5:
            continue
        label = LABELS[cls]
        det_map.setdefault(label, []).append({
            "bbox": (x1, y1, x2, y2),
            "conf": conf
        })

    # 2. Para cada par EPP, decidir cuál dibujar
    for positive, negative in [
        ("goggles", "no-goggles"),
        ("helmet", "no-helmet"),
        ("vest",    "no-vest")
    ]:
        if negative in det_map:
            # Tomamos la detección 'no-xxx' de mayor confianza
            choice = max(det_map[negative], key=lambda d: d["conf"])
            label = negative
        elif positive in det_map:
            choice = max(det_map[positive], key=lambda d: d["conf"])
            label = positive
        else:
            continue  # ni sí ni no: nada que dibujar para este par

        x1, y1, x2, y2 = choice["bbox"]
        color = (0, 0, 255) if label.startswith("no-") else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"{label} {choice['conf']:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1
        )

    # 3. Mostrar también 'Person' (puede haber varias)
    if "Person" in det_map:
        for det in det_map["Person"]:
            x1, y1, x2, y2 = det["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(
                frame,
                f"Person {det['conf']:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
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

        # Inferencia
        results = model(frame)[0]
        
        # Dibujar detecciones
        draw_detections(frame, results)

        # Mostrar
        cv2.imshow("Detección EPP (YOLOv11)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Demo interrumpido por el usuario.")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
