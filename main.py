import cv2
from detector import Detector
from tracker import Tracker
from drawing import draw_tracks
from config import DIST_THRESHOLD, MISS_THRESHOLD, CONFIDENCE_THRESHOLD

detector = Detector("yolo26n.pt")
tracker = Tracker()

cap = cv2.VideoCapture(0)   # 0 = default camera
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        new_elements = detector.feed_frame(frame)
        tracker.update_demo(new_elements)

        out = draw_tracks(frame, tracker.tracks)
        cv2.imshow("cam", out)          # optional preview
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()