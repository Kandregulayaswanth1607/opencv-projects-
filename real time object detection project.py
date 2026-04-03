import cv2
import numpy as np

cap=cv2.VideoCapture(0,cv2.CAP_MSMF)
if not cap.isOpened():
    print("cannont open the webcam")
    exit()

while True:
    ret,frame=cap.read()
    if not ret:
      print("cannot recive the frame in the stream")
      continue
      
      

    
    

    grayscale=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    edged=cv2.Canny(grayscale,30,200)
   
    rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

    lower_red1=(0,120,70)
    upper_red1=(10,255,255)
    lower_red2=(170,120,70)
    upper_red2=(180,255,255)
    mask1=cv2.inRange(rgb,lower_red1,upper_red1)
    mask2=cv2.inRange(rgb,lower_red2,upper_red2)
    mask=cv2.add(mask1,mask2)
    result=cv2.bitwise_and(frame,frame,mask=mask)

    contours,heir=cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
       if cv2.contourArea(cnt) < 500:
        continue

       x, y, w, h = cv2.boundingRect(cnt)

    # Draw box
       cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Add label
       cv2.putText(frame, "Object", (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 0), 2)
       
    cv2.imshow("contours", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    

   
cap.release()
cv2.destroyAllWindows()