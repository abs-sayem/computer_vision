import cv2

# Image Capture
video = cv2.VideoCapture(0)
face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
iter = 0
face_data = []
while True:
    ret,frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detected_face = face.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in detected_face:
        crop_img = frame[y:y+h, x:x+w, :]   # crop the face
        resized_img = cv2.resize(crop_img, (50,50))     # resize the frame
        face_data.append(resized_img)   # save to list
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 1)
    cv2.imshow("Frame", frame)
    k = cv2.waitKey(1)
    if k==ord('q') or len(face_data)==50: break
video.release()
cv2.destroyAllWindows()