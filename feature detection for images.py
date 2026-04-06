import cv2
import numpy as np


img1 = cv2.imread(r"C:\Users\USER\OneDrive\Desktop\right side of room.jpeg")
img2 = cv2.imread(r"C:\Users\USER\OneDrive\Desktop\left side of room.jpeg")
img1r=cv2.resize(img1,(640,480))
img2r=cv2.resize(img2,(640,480))
orb = cv2.ORB_create()
vstack=np.vstack((img1,img2))

prev_kp, prev_des = orb.detectAndCompute(img1r, None)
kp, des = orb.detectAndCompute(img2r, None)

if prev_des is not None and des is not None:

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    matches = bf.match(prev_des, des)
    matches = sorted(matches, key=lambda x: x.distance)

    result = cv2.drawMatches(img1r, prev_kp,
                             img2r, kp,
                             matches[:30], None,
                             flags=2)
    
    
    
    cv2.imshow("Feature Matching", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
