import cv2
import numpy as np

def sketch(image):
    
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussainBlur(gray,(5,5),0)
    edges = cv2.Canny(gray_blur,10,70)
    _,thresh = cv2.threshold(edges,120,255,cv2.THRESH_BINARY_INV)
    
    return thresh

cap = cv2.VideoCapture()

while(True):
    _,frame = cap.read()
    edges = sketch(frame)
    cv2.imshow("Live Sketching",edges)
    if cv2.waitKey(0) & 0xFF == ord('q'):
		break
    
cap.release()
cv2.destryAllWindows()
    
    
    
    
    
    
    
    