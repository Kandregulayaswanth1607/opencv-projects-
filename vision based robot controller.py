# version two of the path tracking object project , additionally it will show the arrowed line towards the object.
import cv2
import numpy as np

cap = cv2.VideoCapture(0, cv2.CAP_MSMF)

if not cap.isOpened():
    print("Cannot open webcam")
    exit()

points = []
prev_x, prev_y = 0, 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Cannot receive frame")
        break


    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_blue1 = np.array([100, 140, 50])
    upper_blue1 = np.array([120, 255, 255])
    lower_blue2 = np.array([121, 150, 50])
    upper_blue2 = np.array([140, 255, 255])
    mask1 = cv2.inRange(hsv, lower_blue1, upper_blue1)
    mask2 = cv2.inRange(hsv, lower_blue2, upper_blue2)
    mask = cv2.add(mask1, mask2)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        cnt = max(contours, key=cv2.contourArea)

        if cv2.contourArea(cnt) > 500:
            x, y, w, h = cv2.boundingRect(cnt)
            cx = x + w // 2
            cy = y + h // 2
            # new line added , for the arrowed line
            cv2.arrowedLine(frame, (prev_x, prev_y), (cx, cy), (0, 0, 255), 3)
            alpha = 0.2
            cx = int(alpha * cx + (1 - alpha) * prev_x)
            cy = int(alpha * cy + (1 - alpha) * prev_y)
            dx = cx - prev_x

            if dx > 10:
              direction = "RIGHT"
            elif dx < -10:
              direction = "LEFT"
            else:
             direction = "STABLE"
            dy = cy - prev_y
            

            if abs(dx) > 15 or abs(dy) > 15:
                points.append((cx, cy))
                prev_x, prev_y = cx,cy
            for i in range(1, len(points)):
                cv2.line(frame, points[i-1], points[i], (0, 0, 255), 2)
            
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    
    cv2.imshow("object tracking", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
