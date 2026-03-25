import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

model = YOLO("runs/detect/train/weights/best.pt")
tracker = DeepSort(max_age=30)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]

    detections = []

    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        cls = int(box.cls[0])

        detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls))

    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        l, t, w, h = track.to_ltrb()

        cv2.rectangle(frame, (int(l), int(t)), (int(l+w), int(t+h)), (0,255,0), 2)
        cv2.putText(frame, f"ID {track_id}", (int(l), int(t-10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.imshow("Tracking", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()