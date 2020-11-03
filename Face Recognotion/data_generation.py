import cv2
import numpy as np

face_xml = ''
face_clf = cv2.CascadeClassifier(face_xml)

def get_face(image):
    gray_img = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    faces = face_clf.detectMultiScale(gray_img,1.3,5)
    
    if faces is ():
        return None
    
    else:
        for (x,y,w,h) in faces:
            cropped_face = image[y:y+h,x:x+h]
            
    return (x,y,w,h), cropped_face

cap = cv2.VideoCapture()
num_faces = 0
new_dim = (20,20)

while True:
    _,frame = cap.read()
    if get_face(frame) is not None:
        num_faces = num_faces+1
        (x,y,w,h), face = get_face(frame)
        face = cv2.resize(face,new_dim)
        face = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
        
        path = 'train_data' + str(num_faces) + '.jpg'
        cv2.imwrite(path,face)
        
        cv2.putText(face,str(num_faces),(50,50),cv2.FONT_ITALIC,1,(0,255,0),2)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,0),2)
        cv2.imshow(frame)
        cv2.imshow('Found_face',face)
        
    else:
        print("No face found")
        pass
    
    if num_faces==50:
        break
    
cap.release()
cv2.destroyAllWindows()        
        
        
        
        
        
        
        
        
        

