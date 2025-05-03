"""
Demo de detección de EPP con modelo YOLOv11 + threading (corregido)
"""
import cv2
import threading
import queue
import sys
from pathlib import Path
from ultralytics import YOLO

# Etiquetas del modelo
LABELS = ['Person', 'goggles', 'helmet', 'no-goggles', 'no-helmet', 'no-vest', 'vest']


class FrameGrabber(threading.Thread):
    """Hilo productor: captura frames del video/cámara y los pone en una cola."""
    def __init__(self, src, queue_size=1):
        super().__init__(daemon=True)
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise IOError(f"No se pudo abrir fuente de video: {src}")
        self.queue = queue.Queue(maxsize=queue_size)
        self.stopped = False

    def run(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                # fin de video o error
                self.stopped = True
                break
            # si la cola está llena, descartamos el frame más antiguo
            if self.queue.full():
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    pass
            self.queue.put(frame)

    def read(self, timeout=1):
        """
        Obtiene el frame más reciente.
        Lanza queue.Empty si no hay frame en 'timeout' segundos.
        """
        return self.queue.get(timeout=timeout)

    def release(self):
        """Para el hilo y libera la captura."""
        self.stopped = True
        self.cap.release()


def draw_detections(frame, detections):
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
    return frame


def main():
    # Fuente de video: argumento o archivo por defecto
    video_src = sys.argv[1] if len(sys.argv) > 1 else "epp.mp4"

    # Inicializa y arranca el hilo capturador
    grabber = FrameGrabber(video_src)
    grabber.start()

    # Carga del modelo YOLO
    model_path = Path(__file__).parent / "model" / "best.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo no encontrado en: {model_path}")
    model = YOLO(str(model_path))

    print("Iniciando demo. Pulsa 'q' para salir.")
    while True:
        # Si el hilo terminó y la cola está vacía, salimos
        if grabber.stopped and grabber.queue.empty():
            break
        try:
            frame = grabber.read(timeout=1)
        except queue.Empty:
            continue  # sin frame disponible, reintentar

        # Reducir resolución para acelerar (opcional)
        frame_small = cv2.resize(frame, (0, 0), fx=0.7, fy=0.7)

        # Inferencia (detección de EPP)
        results = model(frame_small)[0]

        # Dibujar resultados
        output = draw_detections(frame_small, results)

        # Mostrar
        cv2.imshow("Detección EPP (YOLOv11)", output)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Limpieza
    grabber.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
