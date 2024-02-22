import cv2
import pickle
import numpy as np
import os
#import PIL.Image   # for showing images, alternative of cv2.imshow(), needs "Pillow" to be installed

# Image Capture
video = cv2.VideoCapture(0)
face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
iter = 0
face_data = []
name = input("Enter your Name: ")

while True:
    ret,frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detected_face = face.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in detected_face:
        crop_img = frame[y:y+h, x:x+w, :]   # crop the face
        resized_img = cv2.resize(crop_img, (50,50))     # resize the frame
        if len(face_data) <= 20 and iter%5 == 0:
            face_data.append(resized_img)   # save to list
        iter += 1
        cv2.putText(frame, str(len(face_data)), (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 1)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 1)
    cv2.imshow("Frame", frame)
    k = cv2.waitKey(1)
    if k==ord('q') or len(face_data)==20: break
video.release()
cv2.destroyAllWindows()

face_data_array = np.asarray(face_data)
reshaped_face_data = face_data_array.reshape(20, -1)

# pickle names
if 'name.pkl' not in os.listdir('data/'):
    names = [name]*20
    with open('data/names.pkl', 'wb') as f: pickle.dump(names, f)
else:
    with open('data/names.pkl', 'rb') as f: names = pickle.load(f)
    names = names + [name]*20
    with open('data/names.pkl', 'wb') as f: pickle.dump(names, f)
    
# pickle images
if 'face_data.pkl' not in os.listdir('data/'):
    with open('data/face_data.pkl', 'wb') as f: pickle.dump(face_data_array, f)
else:
    with open('data/face_data.pkl', 'rb') as f: faces = pickle.load(f)
    faces = np.append(faces, face_data_array, axes=0)
    with open('data/face_data.pkl', 'wb') as f: pickle.dump(faces, f)