from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import numpy as np
import os
#import PIL.Image   # for showing images, alternative of cv2.imshow(), needs "Pillow" to be installed

# Image Capture
video = cv2.VideoCapture(0)
face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# with open('data/abs_sayem_names.pkl', 'rb') as f: LABELS = pickle.load(f)
# with open('data/abs_sayem_face_data.pkl', 'rb') as f: FACES = pickle.load(f)
# Get a list of all files in the data directory
data_dir = 'data'
data_files = [f for f in os.listdir(data_dir) if f.endswith('_names.pkl')]

# Initialize lists to store labels and faces
all_labels = []
all_faces = []

# Iterate through each data file
for file in data_files:
    # Form the file paths for labels and faces
    labels_file = os.path.join(data_dir, file)
    faces_file = os.path.join(data_dir, file.replace('_names.pkl', '_face_data.pkl'))

    # Open and load labels data
    with open(labels_file, 'rb') as f_labels:
        labels_data = pickle.load(f_labels)
        all_labels.extend(labels_data)

    # Open and load face data
    with open(faces_file, 'rb') as f_faces:
        faces_data = pickle.load(f_faces)
        all_faces.extend(faces_data)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(all_faces, all_labels)

while True:
    ret,frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detected_face = face.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in detected_face:
        crop_img = frame[y:y+h, x:x+w, :]   # crop the face
        resized_img = cv2.resize(crop_img, (50,50)).flatten().reshape(1,-1)     # resize the frame
        output = knn.predict(resized_img)
        cv2.putText(frame, str(output[0]), (x,y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2)        
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 1)
    cv2.imshow("Frame", frame)
    k = cv2.waitKey(1)
    if k==ord('q'): break
video.release()
cv2.destroyAllWindows()

