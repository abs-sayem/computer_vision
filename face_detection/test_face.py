from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import numpy as np
import os
#import PIL.Image   # for showing images, alternative of cv2.imshow(), needs "Pillow" to be installed

# Image Capture
video = cv2.VideoCapture(0)
face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
with open('data/names.pkl', 'rb') as f: LABELS = pickle.load(f)
with open('data/face_data.pkl', 'rb') as f: FACES = pickle.load(f)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

while True:
    ret,frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detected_face = face.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in detected_face:
        crop_img = frame[y:y+h, x:x+w, :]   # crop the face
        resized_img = cv2.resize(crop_img, (50,50)).flatten().reshape(1,-1)     # resize the frame
        output = knn.predict(resized_img)
        cv2.putText(frame, str(output[0]), (x,y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 1)        
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 1)
    cv2.imshow("Frame", frame)
    k = cv2.waitKey(1)
    if k==ord('q'): break
video.release()
cv2.destroyAllWindows()

