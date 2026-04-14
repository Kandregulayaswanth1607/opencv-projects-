import cv2
from ultralytics import YOLO
model = YOLO("yolov8n.pt")  
cap = cv2.VideoCapture(0, cv2.CAP_FFMPEG)

if not cap.isOpened():
    print(" Webcam not found!")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame, verbose=False)
    for result in results:
        boxes = result.boxes

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0]) * 100
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            if confidence > 50:
                cv2.rectangle(frame,
                    (x1, y1), (x2, y2),
                    (0, 255, 0), 2)
                label = f"{class_name}: {confidence:.1f}%"
                (w, h), _ = cv2.getTextSize(
                    label,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, 2
                )
                cv2.rectangle(frame,
                    (x1, y1 - 25),
                    (x1 + w, y1),
                    (0, 255, 0), -1)
                cv2.putText(frame, label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 0), 2)
    count = len(results[0].boxes)
    cv2.putText(frame,
        f"Objects detected: {count}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8, (0, 255, 255), 2)

    cv2.putText(frame, "Press ESC to quit",
        (10, frame.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5, (200, 200, 200), 1)

    cv2.imshow("YOLOv8 Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
