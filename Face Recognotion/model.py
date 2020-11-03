import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

train_dir = 'train_data/'
files = [image for image in listdir(path)]

images = []
targets = []

for i,file in enumerate(files):
    image_path = os.path.join(path,file)
    image = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
    image = np.asarray(image,dtype=np.uint8)
    images.append(image)
    targets.append(i)
    
targets = np.asarray(targets,dtype=np.uint8)
clf = cv2.face.createLBPHFaceRecognizer()	
print('Start Training')
clf.train(inputs,targets)
print('Training Done')

face_xml = ''
face_clf = cv2.CascadeClassifier(face_xml)

def find_face(image):
    gray_img = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    faces = face_clf.detectMultiScale(gray_img,1.3,5)
    
    if faces is ():
        return image
    
    for (x,y,w,h) in faces:
        detected_face = image[y:y+h,x:x+w]
        detected_face = cv2.resize(detected_face,(200,200))
        
    return detected_face

cap = cv2.VideoCapture()

while True:
    _,frame = cap.read()
    face = find_face(frame)
    try:
        gray_face = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
        label,score = clf.predict(gray_face)
        
        if score < 500:
			confidence_score = int(100 * (1 - score/400))
			message = 'I am {} confident '.format(str(confidence_score))

		cv2.putText(frame, message, (50,50), cv2.FONT_ITALIC, 1, (200,0,200), 2)

		if score > 75:
			message = 'Sorry! You are not who you say'
			cv2.putText(frame, message, (150,50), cv2.FONT_ITALIC, 1, (200,200,0), 2)
			cv2.imshow('I rekognize You', image)
		else:
			message = 'You you are finally here'
			cv2.putText(frame, message, (150,50), cv2.FONT_ITALIC, 1, (0,200,200), 2)	
			cv2.imshow('I rekognize You', image)
	
	except:
		message = 'No face found!'
		cv2.putText(frame, message, (150,150), cv2.FONT_ITALIC, 1, (0,200,200), 2)	
		cv2.imshow('I rekognize You', frame)
		pass
    
    if cv2.waitKey(0):
        break
    
cap.release()
cv2.destroyAllWindows()














