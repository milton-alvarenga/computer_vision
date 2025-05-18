import cv2
import mediapipe as mp

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Use FaceDetection with default model and confidence threshold
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_detection.process(img_rgb)

    if results.detections:
        for detection in results.detections:
            # Draw bounding box and keypoints using MediaPipe utility
            mp_drawing.draw_detection(img, detection)

            # Extract keypoints for left and right eyes
            keypoints = detection.location_data.relative_keypoints
            h, w, _ = img.shape

            # Left eye
            left_eye = keypoints[0]  # index 0 is left eye
            left_eye_coords = (int(left_eye.x * w), int(left_eye.y * h))
            cv2.circle(img, left_eye_coords, 5, (0, 255, 0), cv2.FILLED)

            # Right eye
            right_eye = keypoints[1]  # index 1 is right eye
            right_eye_coords = (int(right_eye.x * w), int(right_eye.y * h))
            cv2.circle(img, right_eye_coords, 5, (0, 255, 0), cv2.FILLED)

    cv2.imshow("Face and Eyes Detection", img)
    if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to exit
        break

cap.release()
cv2.destroyAllWindows()

