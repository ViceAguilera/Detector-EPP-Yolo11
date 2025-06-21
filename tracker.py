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
# Lógica de tracking con ByteTrack
# ----------------------------------------------------------------------------

class ByteTrackWrapper:
    """Tracker usando ByteTrack para detección multi-objeto."""
    def __init__(self, frame_rate: int = 30, track_thresh: float = 0.5):
        if not BYTETRACK_AVAILABLE:
            raise ImportError("BYTETracker no disponible. Instálalo con: pip install cjm_byte_track")
        self.tracker = BYTETracker(frame_rate=frame_rate, track_thresh=track_thresh)

    def update(self, detections, frame):
        if not detections:
            return []
        
        # Convertir detecciones a formato requerido por BYTETracker
        # Formato: [x1, y1, x2, y2, score]
        dets_array = []
        for d in detections:
            x1, y1, x2, y2 = d['bbox']
            dets_array.append([x1, y1, x2, y2, d['conf']])
        
        # Convertir a numpy array
        output_results = np.array(dets_array, dtype=np.float32)
        
        # Obtener dimensiones del frame
        img_height, img_width = frame.shape[:2]
        img_size = (img_width, img_height)
        
        # Actualizar tracker con la API correcta
        tracks = self.tracker.update(
            output_results=output_results,
            img_info=img_size,
            img_size=img_size
        )
        
        # Procesar resultados del tracker
        tracked = []
        for track in tracks:
            x1, y1, x2, y2, track_id = track.tlbr[0], track.tlbr[1], track.tlbr[2], track.tlbr[3], track.track_id
            score = track.score
            
            # Encontrar la etiqueta original más cercana
            for d in detections:
                bx1, by1, bx2, by2 = d['bbox']
                # Comparativa simple de coordenadas
                if abs(bx1 - x1) < 10 and abs(by1 - y1) < 10:
                    tracked.append({
                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                        'label': d['label'],
                        'conf': score,
                        'track_id': int(track_id)
                    })
                    break
        return tracked

# ----------------------------------------------------------------------------
# Dibujo + EPP por persona
# ----------------------------------------------------------------------------

def draw_tracked(frame, tracked_dets):
    for det in tracked_dets:
        x1, y1, x2, y2 = det['bbox']
        label, conf = det['label'], det['conf']
        tid = det['track_id']
        color = (255,255,0) if label=='Person' else ((0,255,0) if not label.startswith('no-') else (0,0,255))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"ID{tid} {label} {conf:.2f}", (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return frame

EPP_PAIRS = [
    ('goggles','no-goggles'),
    ('helmet','no-helmet'),
    ('vest','no-vest'),
]

def summarize_persons(tracked_dets):
    # Separar personas y EPP
    personas = []
    epps = []
    
    for det in tracked_dets:
        if det['label'] == 'Person':
            personas.append(det)
        else:
            epps.append(det)
    
    print(f"\n=== DEBUG: Encontradas {len(personas)} personas y {len(epps)} EPP ===")
    
    # Crear diccionario de personas
    persons = {}
    for idx, persona in enumerate(personas):
        # Usar índice como ID para evitar problemas con track_ids
        persons[idx] = {}
        print(f"Persona {idx}: ID original {persona['track_id']}")
    
    # Asociar cada EPP con la persona más cercana
    for epp in epps:
        epp_x1, epp_y1, epp_x2, epp_y2 = epp['bbox']
        epp_center_x = (epp_x1 + epp_x2) / 2
        epp_center_y = (epp_y1 + epp_y2) / 2
        
        min_distance = float('inf')
        closest_person_idx = None
        
        for idx, persona in enumerate(personas):
            p_x1, p_y1, p_x2, p_y2 = persona['bbox']
            p_center_x = (p_x1 + p_x2) / 2
            p_center_y = (p_y1 + p_y2) / 2
            
            distance = ((epp_center_x - p_center_x) ** 2 + 
                       (epp_center_y - p_center_y) ** 2) ** 0.5
            
            if distance < min_distance:
                min_distance = distance
                closest_person_idx = idx
        
        # Asignar EPP a la persona más cercana si está dentro de un umbral razonable
        if closest_person_idx is not None and min_distance < 300:  # umbral de 300 píxeles
            persons[closest_person_idx][epp['label']] = epp['conf']
            print(f"Asignando {epp['label']} (conf: {epp['conf']:.2f}) a persona {closest_person_idx} (distancia: {min_distance:.1f})")
        else:
            print(f"EPP {epp['label']} muy lejos de cualquier persona (distancia mínima: {min_distance:.1f})")
    
    lines = [f"Detectadas: {len(persons)} persona(s)\n"]
    for idx, (person_idx, items) in enumerate(sorted(persons.items()), start=0):
        lines.append(f"Persona {idx+1}:")
        for pos, neg in EPP_PAIRS:
            if items.get(neg) is not None:
                lines.append(f"  {neg}: {items[neg]:.2f}")
            elif items.get(pos) is not None:
                lines.append(f"  {pos}: {items[pos]:.2f}")
            else:
                lines.append(f"  {pos}: no detectado")
        lines.append("")
    return "\n".join(lines)
