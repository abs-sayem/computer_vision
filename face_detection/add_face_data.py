import cv2

# Image Capture
video = cv2.VideoCapture(0)
faces = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
while True:
    ret,frame = video.read()
    cv2.imshow("Frame", frame)
    k = cv2.waitKey(1)
    if k == ord('q'): break
video.release()
cv2.destroyAllWindows()