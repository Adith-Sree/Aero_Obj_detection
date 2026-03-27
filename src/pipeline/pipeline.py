import cv2
from src.detection.yolo_detector import Detector
from src.tracking.tracker import Tracker
from src.reid.reid import ReIdentifier
from src.behavior.behavior import BehaviorAnalyzer

class UAVPipeline:
    def __init__(self):
        print("INIT CALLED 🔥")

        self.detector = Detector("models/weights/best.pt")
        self.tracker = Tracker()
        self.reid = ReIdentifier()
        self.behavior = BehaviorAnalyzer()

        self.flagged_ids = set()

    def run(self):
        cap = cv2.VideoCapture(0)

        print("Press 'f' to flag | 'q' to quit")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = self.detector.detect(frame)

            detections = []
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                detections.append(([x1, y1, x2-x1, y2-y1], conf, cls))

            tracks = self.tracker.update(detections, frame)

            for track in tracks:
                if not track.is_confirmed():
                    continue

                track_id = track.track_id
                l, t, w, h = track.to_ltrb()

                center = (int(l + w/2), int(t + h/2))
                self.behavior.update(track_id, center)

                feature = self.reid.extract_feature(frame, l, t, w, h)

                # Draw box
                cv2.rectangle(frame, (int(l), int(t)), (int(l+w), int(t+h)), (0,255,0), 2)
                cv2.putText(frame, f"ID {track_id}", (int(l), int(t-10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

                # Flag alert
                if track_id in self.flagged_ids:
                    cv2.putText(frame, "FLAGGED", (int(l), int(t-30)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                    print(f"🚨 ALERT: Object {track_id}")

                # ReID
                if feature is not None and self.reid.match(feature):
                    cv2.putText(frame, "RE-ID", (int(l), int(t-50)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

                # Stationary
                if self.behavior.is_stationary(track_id):
                    cv2.putText(frame, "STOPPED", (int(l), int(t-70)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

            cv2.imshow("UAV System", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('f'):
                for track in tracks:
                    if track.is_confirmed():
                        self.flagged_ids.add(track.track_id)

                        l, t, w, h = track.to_ltrb()
                        feature = self.reid.extract_feature(frame, l, t, w, h)
                        if feature is not None:
                            self.reid.add(feature)

            if key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()