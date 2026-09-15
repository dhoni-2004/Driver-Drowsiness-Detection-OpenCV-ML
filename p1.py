import cv2
import numpy as np
import time
import os

# Load Cascade Classifiers
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
mouth_cascade = cv2.CascadeClassifier("haarcascade_smile.xml")

cap = cv2.VideoCapture(0)

# Detection Counters
consecutive_closed = 0
yawn_frames = 0
EYE_CLOSED_LIMIT = 9     # ~0.5 to 0.75 seconds of eye closure
YAWN_LIMIT = 15          # Sustained open mouth frames

# Alert & Metric Tracking
last_alert_time = 0
prev_frame_time = 0
fatigue_score = 0        # 0 to 100 scale

print("Enhanced Driver Monitor Active. Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    h_screen, w_screen, _ = frame.shape
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate real-time FPS
    curr_time = time.time()
    fps = int(1 / (curr_time - prev_frame_time + 1e-6))
    prev_frame_time = curr_time

    # Detect primary face
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(120, 120))

    eye_status_text = "Tracking..."
    mouth_status_text = "Normal"
    status_color = (0, 255, 0)

    if len(faces) > 0:
        (x, y, w, h) = faces[0]

        # Draw Face Bounding Box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 200, 0), 2)

        # 1. Eye Region (Upper 20% to 50% of face)
        eye_roi_gray = gray[y + int(h * 0.20):y + int(h * 0.50), x:x + w]
        eye_roi_color = frame[y + int(h * 0.20):y + int(h * 0.50), x:x + w]
        eyes = eye_cascade.detectMultiScale(eye_roi_gray, scaleFactor=1.1, minNeighbors=4, minSize=(25, 25))

        is_closed = False
        if len(eyes) == 0:
            is_closed = True
        else:
            dark_pixels = 0
            for (ex, ey, ew, eh) in eyes[:2]:
                cv2.rectangle(eye_roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
                eye_crop = eye_roi_gray[ey:ey + eh, ex:ex + ew]
                _, thresh = cv2.threshold(eye_crop, 60, 255, cv2.THRESH_BINARY_INV)
                dark_pixels += np.sum(thresh == 255)

            if len(eyes) > 0 and dark_pixels < 25:
                is_closed = True

        # 2. Mouth Region (Lower 60% to 95% of face)
        mouth_roi_gray = gray[y + int(h * 0.60):y + int(h * 0.95), x:x + w]
        mouth_roi_color = frame[y + int(h * 0.60):y + int(h * 0.95), x:x + w]
        mouths = mouth_cascade.detectMultiScale(mouth_roi_gray, scaleFactor=1.7, minNeighbors=18, minSize=(35, 35))

        is_yawning = False
        if len(mouths) > 0:
            for (mx, my, mw, mh) in mouths[:1]:
                cv2.rectangle(mouth_roi_color, (mx, my), (mx + mw, my + mh), (0, 165, 255), 2)
                if mh > int(h * 0.15):  # Mouth open height exceeds threshold
                    is_yawning = True

        # Update eye state & fatigue score
        if is_closed:
            consecutive_closed += 1
            fatigue_score = min(100, fatigue_score + 3)
            eye_status_text = "Eyes: CLOSED"
        else:
            consecutive_closed = 0
            fatigue_score = max(0, fatigue_score - 1)
            eye_status_text = "Eyes: OPEN"

        # Update yawn state
        if is_yawning:
            yawn_frames += 1
            fatigue_score = min(100, fatigue_score + 2)
            mouth_status_text = "Mouth: YAWNING"
        else:
            yawn_frames = 0
            mouth_status_text = "Mouth: NORMAL"

        # 3. Decision Logic & Alert Triggering
        alert_msg = ""
        now = time.time()

        if consecutive_closed >= EYE_CLOSED_LIMIT:
            alert_msg = "DROWSINESS DETECTED!"
            status_color = (0, 0, 255)
            if now - last_alert_time > 2.0:
                os.system('say "Wake up" &')
                last_alert_time = now

        elif yawn_frames >= YAWN_LIMIT:
            alert_msg = "YAWN DETECTED - TAKE A BREAK"
            status_color = (0, 140, 255)
            if now - last_alert_time > 3.5:
                os.system('say "Driver yawning. Consider resting." &')
                last_alert_time = now

        # Draw red warning banner across screen center if active
        if alert_msg:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, int(h_screen * 0.4)), (w_screen, int(h_screen * 0.55)), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            cv2.putText(frame, alert_msg, (int(w_screen * 0.08), int(h_screen * 0.50)),
                        cv2.FONT_HERSHEY_DUPLEX, 1.1, status_color, 3)

    else:
        eye_status_text = "NO DRIVER DETECTED"
        fatigue_score = max(0, fatigue_score - 1)

    # 4. Heads-Up Display (HUD) Dashboard
    # Background HUD box
    hud_bg = frame.copy()
    cv2.rectangle(hud_bg, (10, 10), (320, 150), (20, 20, 20), -1)
    cv2.addWeighted(hud_bg, 0.7, frame, 0.3, 0, frame)

    # Telemetry Text
    cv2.putText(frame, f"FPS: {fps}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(frame, eye_status_text, (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)
    cv2.putText(frame, mouth_status_text, (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

    # Fatigue Risk Bar
    cv2.putText(frame, f"Risk: {fatigue_score}%", (20, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    bar_color = (0, 255, 0) if fatigue_score < 40 else ((0, 165, 255) if fatigue_score < 75 else (0, 0, 255))
    cv2.rectangle(frame, (110, 117), (300, 131), (70, 70, 70), 1)
    cv2.rectangle(frame, (110, 117), (110 + int(1.9 * fatigue_score), 131), bar_color, -1)

    cv2.imshow("Driver Safety & Fatigue Monitoring System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()