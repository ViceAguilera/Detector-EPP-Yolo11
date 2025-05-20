"""
    Demo de detección de EPP con modelo YOLOv11
"""
import sys
import threading
import cv2
from pathlib import Path
from ui import AppUI
from tracker import IoUTracker, load_model, draw_tracked, summarize_persons


def main(app: AppUI, source: str):
    """Bucle continuo de captura, detección y actualización de UI."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: no se pudo abrir el video o dispositivo {source}")
        return

    model = load_model(Path(__file__).parent / 'model' / 'best.pt')
    tracker = IoUTracker(iou_thresh=0.3, max_lost=5)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Fin del video o error al leer el frame.")
            break

        # Inferencia YOLOv11
        results = model(frame)[0]
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            if conf < 0.5:
                continue
            label = model.names[cls]
            detections.append({'bbox':(x1,y1,x2,y2),'label':label,'conf':conf})

        # Tracking
        tracked = tracker.update(detections)
        annotated = draw_tracked(frame.copy(), tracked)
        
        # Actualizar interfaz
        app.update_frame(annotated)
        resumen = summarize_persons(tracked)
        app.update_tracks(resumen)


    cap.release()


if __name__ == '__main__':
    video_src = sys.argv[1] if len(sys.argv)>1 else 'epp.mp4'
    app = AppUI(width=1366, height=720)
    t = threading.Thread(target=main, args=(app, video_src), daemon=True)
    t.start()
    app.run()