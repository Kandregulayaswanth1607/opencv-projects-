import cv2
import numpy as np
img=cv2.imread(r"C:\Users\USER\Downloads\maneph9-dome-5818580_1920.jpg")
img1=cv2.imread(r"C:\Users\USER\Downloads\maneph9-dome-5818580_1920.jpg")
if img is None:
   print("error:the image is not thier")
else:
   img_resize=cv2.resize(img,(640,480))   
   image_blur=cv2.GaussianBlur(img_resize,(5,5),0)
   edge=cv2.Canny(image_blur,100,200,5,L2gradient=True)
   resized_edge=cv2.resize(edge,(640,480))
   resized_edgeR=cv2.cvtColor(resized_edge,cv2.COLOR_GRAY2RGB)
   result=np.hstack((img_resize,resized_edgeR))
   cv2.imshow("edge detection", result)
   cv2.waitKey(0)
   cv2.destroyAllWindows()