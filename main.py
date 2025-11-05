"""
    Demo de detección de EPP con modelo YOLOv11
"""
import argparse
import math
import threading
from pathlib import Path
from typing import Optional

import cv2

from ui import AppUI
from tracker import (
    ByteTrackWrapper,
    MIN_CONF_THRESH,
    draw_tracked_with_ppe,
    load_model,
    summarize_persons_iou,
)


def _resolve_tracker_fps(cap: cv2.VideoCapture, override: Optional[float]) -> float:
    """Determina el FPS efectivo para el tracker."""
    if override is not None and override > 0:
        return override
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps and math.isfinite(fps) and fps > 1.0:
        return fps
    return 30.0


def main(app: AppUI, source: str, tracker_fps: Optional[float] = None, stop_event: Optional[threading.Event] = None):
    """Bucle continuo de captura, detección y actualización de UI."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: no se pudo abrir el video o dispositivo {source}")
        return

    try:
        model = load_model(Path(__file__).parent / 'model' / 'best.pt')

        effective_fps = _resolve_tracker_fps(cap, tracker_fps)
        frame_rate = max(1, int(round(effective_fps)))
        print(f"Tracker configurado a {frame_rate} FPS (solicitado: {tracker_fps})")
        tracker = ByteTrackWrapper(frame_rate=frame_rate, track_thresh=0.5)

        while True:
            if stop_event is not None and stop_event.is_set():
                break

            ret, frame = cap.read()
            if not ret:
                print("Fin del video o error al leer el frame.")
                if stop_event is not None:
                    stop_event.set()
                app.request_close()
                break

            # Inferencia YOLOv11
            results = model(frame)[0]
            detections = []
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cls = int(box.cls[0])
                label = model.names[cls]
                conf = float(box.conf[0])
                min_conf = MIN_CONF_THRESH.get(label.lower(), MIN_CONF_THRESH['person'])
                if conf < min_conf:
                    continue
                detections.append({'bbox': (x1, y1, x2, y2), 'label': label, 'conf': conf})

            # Tracking con ByteTrack (o fallback)
            tracked = tracker.update(detections, frame)
            annotated = draw_tracked_with_ppe(frame.copy(), tracked, detections)

            if stop_event is not None and stop_event.is_set():
                break

            # Actualizar interfaz
            app.update_frame(annotated)
            resumen = summarize_persons_iou(tracked, detections)
            app.update_tracks(resumen)
    finally:
        cap.release()
        if stop_event is not None:
            stop_event.set()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Demo de detección de EPP con YOLOv11 + ByteTrack")
    parser.add_argument('--source', default='epp.mp4', help="Ruta a video o índice de cámara")
    parser.add_argument('--fps', type=float, default=None, help="FPS forzado para el tracker")
    parser.add_argument('--width', type=int, default=1366, help="Ancho de la ventana UI")
    parser.add_argument('--height', type=int, default=720, help="Alto de la ventana UI")
    args = parser.parse_args()

    stop_event = threading.Event()
    app = AppUI(width=args.width, height=args.height, stop_event=stop_event)
    worker = threading.Thread(target=main, args=(app, args.source, args.fps, stop_event), daemon=False)
    worker.start()
    try:
        app.run()
    finally:
        stop_event.set()
        worker.join()
