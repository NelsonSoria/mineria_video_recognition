import cv2
import torch
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from torchreid import models
from torchreid.utils.torchtools import load_pretrained_weights

# --- CARGAR MODELOS ---
# YOLOv8
yolo_model = YOLO("yolov8n.pt")  # usa yolov8s.pt o yolov8m.pt si quieres más precisión

# Deep SORT
tracker = DeepSort(max_age=30)

# OSNet para re-ID
reid_model = models.build_model(name='osnet_x1_0', num_classes=1000)
load_pretrained_weights(reid_model, 'osnet_x1_0_imagenet.pth')
reid_model.eval().cuda()

# --- CAPTURA WEBCAM ---
cap = cv2.VideoCapture("peoplewalking.mp4")  # Cambia esto por la ruta real de tu video


while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = yolo_model(frame)[0]

    detections = []
    for box in results.boxes:
        if int(box.cls[0]) != 0:  # solo personas
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        detections.append(([x1, y1, x2 - x1, y2 - y1], conf, "person"))

    # Seguimiento
    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        l, t, w, h = map(int, track.to_ltrb())
        cv2.rectangle(frame, (l, t), (w, h), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {track_id}", (l, t - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Detección y seguimiento", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
