"""
    Funciones de carga de modelo, tracking por IoU o StrongSORT y utilidades de dibujo.
"""
from pathlib import Path
from ultralytics import YOLO
import cv2

# Opcional: StrongSORT si está disponible
try:
    from strongsort import StrongSORT
    STRONGSORT_AVAILABLE = True
except ImportError:
    STRONGSORT_AVAILABLE = False

# ----------------------------------------------------------------------------
# Carga de modelo
# ----------------------------------------------------------------------------

def load_model(model_path: Path) -> YOLO:
    """Carga el modelo YOLO desde disco (levanta excepción si falta)."""
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo no encontrado en: {model_path}")
    return YOLO(str(model_path))

# ----------------------------------------------------------------------------
# Utilidades de geometría
# ----------------------------------------------------------------------------

def iou(boxA, boxB):
    """Intersección sobre Unión entre dos bboxes (x1,y1,x2,y2)."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = boxAArea + boxBArea - interArea
    return interArea / union if union else 0.0

# ----------------------------------------------------------------------------
# Lógica de tracking (IoU)
# ----------------------------------------------------------------------------

EPP_PAIRS = [
    ("goggles", "no-goggles"),
    ("helmet", "no-helmet"),
    ("vest", "no-vest"),
]

class Track:
    def __init__(self, track_id: int, bbox):
        self.id = track_id
        self.bbox = bbox
        self.lost = 0

class IoUTracker:
    def __init__(self, iou_thresh: float = 0.3, max_lost: int = 5):
        self.iou_thresh = iou_thresh
        self.max_lost = max_lost
        self.tracks = []
        self.next_id = 0

    @staticmethod
    def _resolve_conflicts(detections):
        if not detections:
            return []
        label_map = {}
        for idx, det in enumerate(detections):
            label_map.setdefault(det["label"], []).append(idx)
        keep = set()
        for pos, neg in EPP_PAIRS:
            if neg in label_map:
                best_idx = max(label_map[neg], key=lambda i: detections[i]["conf"])
                keep.add(best_idx)
            elif pos in label_map:
                best_idx = max(label_map[pos], key=lambda i: detections[i]["conf"])
                keep.add(best_idx)
        if "Person" in label_map:
            keep.update(label_map["Person"])
        return [detections[i] for i in sorted(keep)]

    def update(self, detections):
        detections = self._resolve_conflicts(detections)
        matches, used = {}, set()
        for track in self.tracks:
            best_iou = 0.0
            best_det = None
            for idx, det in enumerate(detections):
                if idx in used:
                    continue
                score = iou(track.bbox, det["bbox"])
                if score > best_iou:
                    best_iou, best_det = score, idx
            if best_iou >= self.iou_thresh:
                matches[track.id] = best_det
                used.add(best_det)
                track.bbox = detections[best_det]["bbox"]
                track.lost = 0
            else:
                track.lost += 1
        self.tracks = [t for t in self.tracks if t.lost <= self.max_lost]
        for idx, det in enumerate(detections):
            if idx not in used:
                self.tracks.append(Track(self.next_id, det["bbox"]))
                matches[self.next_id] = idx
                self.next_id += 1
        for tid, d_idx in matches.items():
            detections[d_idx]["track_id"] = tid
        return detections

# ----------------------------------------------------------------------------
# Tracker con StrongSORT (opcional)
# ----------------------------------------------------------------------------

class StrongSORTTracker:
    def __init__(self, reid_weights_path: str = "osnet_x0_25_market1501.pt", device="cuda"):
        if not STRONGSORT_AVAILABLE:
            raise ImportError("StrongSORT no está disponible. Instálalo con: pip install strongsort")
        self.tracker = StrongSORT(
            model_weights=Path(reid_weights_path),
            device=device,
            fp16=True
        )

    def update(self, detections, frame):
        dets = []
        for det in detections:
            if "bbox" in det and "conf" in det:
                x1, y1, x2, y2 = det["bbox"]
                dets.append([x1, y1, x2, y2, det["conf"]])
        outputs = self.tracker.update(dets, frame)
        tracked = []
        for i, obj in enumerate(outputs):
            x1, y1, x2, y2, track_id = obj
            if i < len(detections):
                det = detections[i].copy()
                det["bbox"] = (x1, y1, x2, y2)
                det["track_id"] = track_id
                tracked.append(det)
        return tracked

# ----------------------------------------------------------------------------
# Dibujo + EPP por persona
# ----------------------------------------------------------------------------

def draw_tracked(frame, tracked_dets):
    for det in tracked_dets:
        x1, y1, x2, y2 = det["bbox"]
        label = det["label"]
        conf = det["conf"]
        tid = det["track_id"]
        if label.startswith("no-"):
            color = (0, 0, 255)
        elif label == "Person":
            color = (255, 255, 0)
        else:
            color = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"ID{tid} {label} {conf:.2f}",
            (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
        )
    return frame

# ----------------------------------------------------------------------------
# Generador de texto resumen para panel Tkinter
# ----------------------------------------------------------------------------

def summarize_persons(tracked_dets):
    summary = {}
    for det in tracked_dets:
        tid = det.get("track_id")
        label = det.get("label")
        conf = det.get("conf")
        if tid is None:
            continue
        if tid not in summary:
            summary[tid] = {
                "helmet": None, "no-helmet": None,
                "goggles": None, "no-goggles": None,
                "vest": None, "no-vest": None
            }
        if label in summary[tid]:
            summary[tid][label] = conf

    # Formatear salida legible
    report = [f"Detectadas: {len(summary)} persona(s)\n"]
    for tid, items in summary.items():
        report.append(f"Person {tid}:")
        for pair in EPP_PAIRS:
            pos, neg = pair
            conf_pos = items[pos]
            conf_neg = items[neg]
            if conf_neg is not None:
                report.append(f"  {neg}: {conf_neg:.2f}")
            elif conf_pos is not None:
                report.append(f"  {pos}: {conf_pos:.2f}")
            else:
                report.append(f"  {pos}: no detectado")
        report.append("")
    return "\n".join(report)
