"""
Privis+  - Screen Privacy Monitor (Enhanced)
Features:
- Detects owner vs watchers with a webcam.
- Shows visual privacy alert when a watcher is detected.
- Plays a short beep sound on alert (Windows-only via winsound).
- Blurs watcher faces for their privacy.
- Keeps a running count of total alerts this session.
"""

import cv2
import time
from collections import deque
import winsound  # built-in on Windows [web:135][web:138]

# ---------- CONFIG ----------
FRONT_FACE_MIN_AREA = 9000
BACK_FACE_MAX_AREA = 8000
STABLE_FRAMES_REQUIRED = 10
ALERT_COOLDOWN_SECONDS = 5

BEEP_FREQ = 1200     # Hz
BEEP_DURATION = 300  # ms

watcher_history = deque(maxlen=STABLE_FRAMES_REQUIRED)
total_alerts = 0

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)  # standard OpenCV face detector [web:126]

cap = cv2.VideoCapture(0)

last_alert_time = 0
alert_active = False


def classify_faces(faces, frame_w, frame_h):
    """
    Decide which face is the laptop owner and which are potential watchers.
    - Owner: largest face (closest to camera).
    - Watcher: smaller face, roughly above/behind the owner (higher in frame).
    """
    owner = None
    watchers = []

    if len(faces) == 0:
        return owner, watchers

    areas = [(w * h) for (x, y, w, h) in faces]
    main_idx = max(range(len(faces)), key=lambda i: areas[i])
    owner = faces[main_idx]

    for i, (x, y, w, h) in enumerate(faces):
        if i == main_idx:
            continue
        area = areas[i]
        center_y = y + h / 2

        if area <= BACK_FACE_MAX_AREA and center_y < frame_h * 0.6:
            watchers.append((x, y, w, h))

    return owner, watchers


def blur_face_region(frame, x, y, w, h):
    """
    Apply strong Gaussian blur to given face region to anonymize it. [web:80][web:123]
    """
    x, y = max(0, x), max(0, y)
    w, h = max(1, w), max(1, h)
    roi = frame[y:y + h, x:x + w]
    if roi.size != 0:
        blurred = cv2.GaussianBlur(roi, (51, 51), 30)
        frame[y:y + h, x:x + w] = blurred


def trigger_alert():
    """
    Activate the alert state and play a beep sound.
    """
    global total_alerts, alert_active, last_alert_time
    total_alerts += 1
    alert_active = True
    last_alert_time = time.time()
    # short beep so user notices the watcher [web:132][web:135][web:138]
    winsound.Beep(BEEP_FREQ, BEEP_DURATION)


print("Privis+ running. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40)
    )  # webcam face detection [web:126]

    owner_face, watcher_faces = classify_faces(faces, w, h)

    # draw owner
    if owner_face is not None:
        (x, y, fw, fh) = owner_face
        cv2.rectangle(frame, (x, y), (x + fw, y + fh), (0, 255, 0), 2)
        cv2.putText(frame, "You", (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # draw + blur watchers
    for (x, y, fw, fh) in watcher_faces:
        blur_face_region(frame, x, y, fw, fh)  # anonymize watcher [web:80][web:123]
        cv2.rectangle(frame, (x, y), (x + fw, y + fh), (0, 0, 255), 2)
        cv2.putText(frame, "Watcher?", (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # update history
    watcher_history.append(1 if len(watcher_faces) > 0 else 0)

    # trigger alert if watcher is stable over several frames
    if sum(watcher_history) == watcher_history.maxlen:
        now = time.time()
        if now - last_alert_time > ALERT_COOLDOWN_SECONDS:
            trigger_alert()

    # auto-hide alert after cooldown
    if time.time() - last_alert_time > ALERT_COOLDOWN_SECONDS:
        alert_active = False

    # transparent alert banner overlay [web:133]
    if alert_active:
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (0, 0),
            (w, int(h * 0.18)),
            (0, 0, 255),
            -1
        )
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        cv2.putText(
            frame,
            "Privacy Alert: Someone may be watching your screen!",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

    # status bar with total alerts
    cv2.rectangle(frame, (0, h - 35), (w, h), (0, 0, 0), -1)
    cv2.putText(
        frame,
        f"Alerts this session: {total_alerts}",
        (10, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2
    )

    cv2.imshow("Privis+ - Screen Privacy Monitor", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
