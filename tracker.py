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

    def update(self, detections):
        persons = [d for d in detections if d["label"] == "Person"]
        others = [d for d in detections if d["label"] != "Person"]

        # Tracking solo de personas
        matches, used = {}, set()
        for track in self.tracks:
            best_iou = 0.0
            best_det = None
            for idx, det in enumerate(persons):
                if idx in used:
                    continue
                score = iou(track.bbox, det["bbox"])
                if score > best_iou:
                    best_iou, best_det = score, idx
            if best_iou >= self.iou_thresh:
                matches[track.id] = best_det
                used.add(best_det)
                track.bbox = persons[best_det]["bbox"]
                track.lost = 0
            else:
                track.lost += 1

        self.tracks = [t for t in self.tracks if t.lost <= self.max_lost]
        for idx, det in enumerate(persons):
            if idx not in used:
                self.tracks.append(Track(self.next_id, det["bbox"]))
                matches[self.next_id] = idx
                self.next_id += 1

        for tid, d_idx in matches.items():
            persons[d_idx]["track_id"] = tid

        # Asociar objetos (helmet, etc.) a personas por IoU
        final_dets = persons.copy()
        for obj in others:
            ox1, oy1, ox2, oy2 = obj["bbox"]
            ocx, ocy = (ox1 + ox2) // 2, (oy1 + oy2) // 2

            best_tid = None
            best_iou = 0.0
            for p in persons:
                if "track_id" not in p:
                    continue
                px1, py1, px2, py2 = p["bbox"]
                margin = 40

                if not (px1 - margin <= ocx <= px2 + margin and py1 - margin <= ocy <= py2 + margin):
                    continue

                score = iou(p["bbox"], obj["bbox"])
                if score > best_iou:
                    best_iou = score
                    best_tid = p["track_id"]

            if best_tid is not None and best_iou > 0.07:
                obj["track_id"] = best_tid
                final_dets.append(obj)

        return final_dets

# ----------------------------------------------------------------------------
# Dibujo + EPP por persona
# ----------------------------------------------------------------------------

def draw_tracked(frame, tracked_dets):
    for det in tracked_dets:
        x1, y1, x2, y2 = det["bbox"]
        label = det["label"]
        conf = det["conf"]
        tid = det.get("track_id", -1)
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
# Resumen sin ID visibles
# ----------------------------------------------------------------------------

def summarize_persons(tracked_dets):
    personas = {}
    for det in tracked_dets:
        tid = det.get("track_id")
        label = det["label"]
        conf = det["conf"]
        if tid is None or label == "Person":
            continue
        if tid not in personas:
            personas[tid] = {
                "helmet": None, "no-helmet": None,
                "goggles": None, "no-goggles": None,
                "vest": None, "no-vest": None
            }
        if label in personas[tid]:
            personas[tid][label] = conf
    
    output = [f"Detectadas: {len(personas)} persona(s)\n"]
    for idx, (_, estado) in enumerate(personas.items(), 1):
        output.append(f"Persona {idx}:")
        for pos, neg in EPP_PAIRS:
            if estado[neg] is not None:
                output.append(f"  {neg}: {estado[neg]:.2f}")
            elif estado[pos] is not None:
                output.append(f"  {pos}: {estado[pos]:.2f}")
            else:
                output.append(f"  {pos}: no detectado")
        output.append("")
    return "\n".join(output)