"""
    Funciones de carga de modelo, tracking con ByteTrack y utilidades de dibujo.
"""
from pathlib import Path
from ultralytics import YOLO
import cv2
import numpy as np

# Verificar disponibilidad de ByteTrack
try:
    from cjm_byte_track.core import BYTETracker
    BYTETRACK_AVAILABLE = True
except ImportError:
    BYTETRACK_AVAILABLE = False

# ----------------------------------------------------------------------------
# Carga de modelo
# ----------------------------------------------------------------------------

def load_model(model_path: Path) -> YOLO:
    """Carga el modelo YOLO desde disco (levanta excepción si falta)."""
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo no encontrado en: {model_path}")
    return YOLO(str(model_path))

# ----------------------------------------------------------------------------
# Utilidades geométricas
# ----------------------------------------------------------------------------
def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0

def _rect_area(b):
    x1, y1, x2, y2 = b
    return max(0, x2 - x1) * max(0, y2 - y1)

def _rect_intersection(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1, y1 = max(ax1, bx1), max(ay1, by1)
    x2, y2 = min(ax2, bx2), min(ay2, by2)
    return max(0, x2 - x1) * max(0, y2 - y1)

def _region_from_person(person_bbox, frac_y1, frac_y2):
    x1, y1, x2, y2 = person_bbox
    h = max(1, y2 - y1)
    ry1 = int(y1 + frac_y1 * h)
    ry2 = int(y1 + min(1.0, frac_y2) * h)
    return (x1, ry1, x2, ry2)

# ----------------------------------------------------------------------------
# Lógica de tracking con ByteTrack (sólo personas)
# ----------------------------------------------------------------------------

class ByteTrackWrapper:
    """Tracker usando ByteTrack para detección de personas."""
    def __init__(self, frame_rate: int = 30, track_thresh: float = 0.5):
        if not BYTETRACK_AVAILABLE:
            raise ImportError("BYTETracker no disponible. Instálalo con: pip install cjm_byte_track")
        self.tracker = BYTETracker(frame_rate=frame_rate, track_thresh=track_thresh)

    def update(self, detections, frame):
        """
        Solo trackea personas:
          - Filtra detecciones a label 'Person'/'person'
          - Empareja tracks con detecciones por IoU
          - Devuelve [{'bbox', 'label', 'conf', 'track_id'}, ...] SOLO de personas
        """
        img_h, img_w = frame.shape[:2]

        # Filtrar SOLO personas (case-insensitive)
        person_dets = [d for d in detections if str(d.get('label', '')).lower() == 'person']

        # Avanzar tracker aunque no haya detecciones (para que caduquen tracks)
        if not person_dets:
            _ = self.tracker.update(
                output_results=np.zeros((0, 5), dtype=np.float32),
                img_info=(img_h, img_w),   # (alto, ancho)
                img_size=(img_h, img_w)    # (alto, ancho)
            )
            return []

        # Convertir a formato requerido por BYTETracker: [x1,y1,x2,y2,score]
        dets_array = np.array(
            [[d['bbox'][0], d['bbox'][1], d['bbox'][2], d['bbox'][3], float(d['conf'])] for d in person_dets],
            dtype=np.float32
        )

        # Actualizar tracker (OJO: orden correcto)
        tracks = self.tracker.update(
            output_results=dets_array,
            img_info=(img_h, img_w),   # (alto, ancho)
            img_size=(img_h, img_w)    # (alto, ancho)
        )

        # Emparejar cada track con la detección de mayor IoU
        tracked = []
        for tr in tracks:
            x1, y1, x2, y2 = int(tr.tlbr[0]), int(tr.tlbr[1]), int(tr.tlbr[2]), int(tr.tlbr[3])
            tid = int(tr.track_id)

            best = None
            best_iou = -1.0
            for d in person_dets:
                iou = iou_xyxy((x1, y1, x2, y2), d['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best = d

            if best is not None and best_iou >= 0.1:
                label = 'Person'
                conf = float(best['conf'])
            else:
                label = 'Person'
                conf = float(getattr(tr, 'score', 0.0))

            tracked.append({
                'bbox': (x1, y1, x2, y2),
                'label': label,
                'conf': conf,
                'track_id': tid
            })
        return tracked

# ----------------------------------------------------------------------------
# Dibujo (personas + PPE)
# ----------------------------------------------------------------------------

_PPE_COLORS = {
    'helmet':  (0, 255, 0),    # verde
    'vest':    (0, 200, 255),  # amarillo/cyan
    'goggles': (255, 0, 255),  # magenta
}

def draw_tracked_with_ppe(frame, tracked_persons, all_dets, show_labels=True):
    """
    Dibuja personas (trackeadas) + PPE (detecciones crudas del frame).
    - tracked_persons: salida de ByteTrackWrapper.update() (solo 'Person')
    - all_dets: lista [{'bbox','label','conf'}, ...] (personas + PPE)
    """
    # 1) Personas (amarillo)
    for det in tracked_persons:
        x1, y1, x2, y2 = det['bbox']
        tid = det['track_id']
        conf = det.get('conf', 0.0)
        color = (255, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"ID{tid} Person {conf:.2f}", (x1, max(10, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # 2) PPE (colores por clase)
    for d in all_dets:
        lbl = str(d.get('label', '')).lower()
        if lbl in _PPE_COLORS:
            x1, y1, x2, y2 = d['bbox']
            color = _PPE_COLORS[lbl]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            if show_labels:
                cv2.putText(frame, f"{lbl} {d.get('conf', 0.0):.2f}", (x1, max(10, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return frame


# ----------------------------------------------------------------------------
# Asociación PPE ↔ Persona (IoU + contención por región)
# ----------------------------------------------------------------------------

# Regiones verticales relativas dentro del bbox de persona (y ∈ [y1 + frac1*h, y1 + frac2*h])
_PPE_REGIONS = {
    'helmet':  (0.00, 0.35),  # cabeza alta
    'goggles': (0.15, 0.45),  # ojos
    'vest':    (0.35, 0.85),  # torso
}
_MIN_PERSON_H   = 80       # px
_IOU_MIN        = 0.10     # IoU mínimo PPE↔persona
_CONTAIN_THRESH = 0.30     # % del área PPE dentro de la región anatómica
_CLASS_THRESH   = {        # conf mínima por clase PPE
    'helmet':  0.45,
    'goggles': 0.30,
    'vest':    0.40,
}

def summarize_persons_iou(tracked_persons, all_dets):
    """
    Resumen por persona usando asociación PPE↔persona por IoU + contención en región.
    - tracked_persons: salida de ByteTrackWrapper.update() (solo 'Person')
    - all_dets: lista de detecciones crudas del frame (personas + PPE) {'bbox','label','conf'}
    Devuelve un string para la UI.
    """
    # Agrupar solo PPE (labels case-insensitive)
    ppe_by_type = {'helmet': [], 'goggles': [], 'vest': []}
    for d in all_dets:
        lbl = str(d.get('label', '')).lower().strip()
        if lbl in ppe_by_type:
            ppe_by_type[lbl].append(d)

    lines = [f"Detectadas: {len(tracked_persons)} persona(s)\n"]

    for idx, person in enumerate(tracked_persons, start=1):
        pid = int(person.get('track_id', -1))
        px1, py1, px2, py2 = person['bbox']
        ph = max(1, py2 - py1)
        lines.append(f"Persona {idx} (ID {pid}):")

        # Si la persona es muy pequeña, evitamos evaluar PPE
        if ph < _MIN_PERSON_H:
            lines.append("  helmet: no evaluado (persona pequeña)")
            lines.append("  goggles: no evaluado (persona pequeña)")
            lines.append("  vest: no evaluado (persona pequeña)")
            lines.append("")
            continue

        # Regiones anatómicas
        regions = {k: _region_from_person(person['bbox'], *_PPE_REGIONS[k]) for k in ppe_by_type.keys()}

        # Por cada PPE, escoger el mejor candidato
        for cls in ('helmet', 'vest', 'goggles'):
            best_score = -1.0
            best_conf = 0.0
            region = regions[cls]

            for epp in ppe_by_type[cls]:
                eb = epp['bbox']
                iou = iou_xyxy(eb, person['bbox'])
                if iou < _IOU_MIN:
                    continue

                inter_in_region = _rect_intersection(eb, region)
                contain_ratio = inter_in_region / max(1, _rect_area(eb))
                if contain_ratio < _CONTAIN_THRESH:
                    continue

                # ranking: contención (70%), IoU (20%), conf (10%)
                score = contain_ratio * 0.7 + iou * 0.2 + float(epp['conf']) * 0.1
                if score > best_score:
                    best_score = score
                    best_conf = float(epp['conf'])

            if best_score >= 0 and best_conf >= _CLASS_THRESH[cls]:
                lines.append(f"  {cls}: {best_conf:.2f}")
            else:
                lines.append(f"  {cls}: no detectado")

        lines.append("")

    return "\n".join(lines)
