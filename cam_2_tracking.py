import cv2
import torch
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from reid_utils import load_reid_model, get_embedding, load_database, is_same_person
from datetime import datetime
import requests

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"[INFO] Usando dispositivo: {device}")

# Cargar modelos
yolo = YOLO("yolov8n.pt")
reid_model = load_reid_model()
db = load_database()

# Inicializar Deep SORT
tracker = DeepSort(max_age=30, n_init=3)

# Captura de cámara o video
video_path = "v1.mp4"
cap = cv2.VideoCapture(0)  # O cambia a tu fuente de cámara

# Almacenar trayectorias (por track_id)
trajectories = {}

# URL servidor para enviar datos (opcional)
SERVER_URL = "http://127.0.0.1:5000/track"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = yolo(frame)[0]

    detections = []
    crops = []

    # Preparar detecciones para Deep SORT (bbox + confidence + clase)
    for box in results.boxes:
        if int(box.cls[0]) != 0:
            continue  # Solo personas
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        bbox = [x1, y1, x2 - x1, y2 - y1]
        detections.append((bbox, conf, "person"))
        crops.append(frame[y1:y2, x1:x2])

    # Actualizar tracker con detecciones
    tracks = tracker.update_tracks(detections, frame=frame)

    for track, crop in zip(tracks, crops):
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        l, t, r, b = map(int, track.to_ltrb())

        # Extraer embedding con ReID para identificar persona (opcional)
        embedding = get_embedding(reid_model, crop)

        # Buscar en base de datos si la persona ya existe
        match_id = None
        for entry in db:
            if is_same_person(entry['embedding'], embedding, threshold=0.75):
                match_id = entry['id']
                break

        # Guardar trayectoria
        cx = int((l + r) / 2)
        cy = int((t + b) / 2)
        if match_id not in trajectories:
            trajectories[match_id] = []
        trajectories[match_id].append((cx, cy))

        # Dibujar bbox, ID y trayectoria
        cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {match_id}", (l, t - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        for i in range(1, len(trajectories[match_id])):
            if trajectories[match_id][i-1] is None or trajectories[match_id][i] is None:
                continue
            cv2.line(frame, trajectories[match_id][i-1], trajectories[match_id][i], (255, 0, 0), 2)

        # Enviar datos al servidor (opcional)
        data = {
            "camera_id": "cam_2",
            "person_id": match_id,
            "x": cx,
            "y": cy,
            "timestamp": datetime.now().isoformat()
        }
        try:
            requests.post(SERVER_URL, json=data, timeout=0.5)
        except requests.exceptions.RequestException:
            print(f"[WARN] Falló el envío de ID {match_id}")

    cv2.imshow("Camara 2 - Seguimiento con Deep SORT + ReID", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
