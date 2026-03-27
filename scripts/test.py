import cv2
import time
import os
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load trained model
model = YOLO("runs/detect/train/weights/best.pt")

# Initialize tracker
tracker = DeepSort(max_age=30)

# Use system camera (0 = default webcam)
cap = cv2.VideoCapture(0)

# Reduce resolution for speed
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Store flagged IDs
flagged_ids = set()

print("Press 'f' to flag objects | 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection
    results = model(frame)[0]

    detections = []

    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        cls = int(box.cls[0])

        detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls))

    # Update tracker
    tracks = tracker.update_tracks(detections, frame=frame)

    # Draw results
    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        l, t, w, h = track.to_ltrb()

        # Draw bounding box
        cv2.rectangle(frame, (int(l), int(t)), (int(l+w), int(t+h)), (0,255,0), 2)

        # Display ID
        cv2.putText(frame, f"ID {track_id}", (int(l), int(t-10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # Check if flagged
        if track_id in flagged_ids:
            cv2.putText(frame, "FLAGGED", (int(l), int(t-30)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

            print(f"🚨 ALERT: Object {track_id} detected!")

            # Mac voice alert (optional)
            os.system('say "Alert detected"')

    # Show frame
    cv2.imshow("UAV Surveillance", frame)

    # Key handling
    key = cv2.waitKey(1) & 0xFF

    # Press 'f' to flag visible objects
    if key == ord('f'):
        for track in tracks:
            if track.is_confirmed():
                flagged_ids.add(track.track_id)
                print(f"Object {track.track_id} flagged ✅")

    # Press 'q' to quit
    if key == ord('q'):
        break

    # ⏳ 5 second delay (IMPORTANT for your camera issue)
    time.sleep(1)

# Cleanup
cap.release()
cv2.destroyAllWindows()