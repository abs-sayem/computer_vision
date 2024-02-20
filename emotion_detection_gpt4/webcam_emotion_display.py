import cv2
from emotion_detection import detect_emotion

def capture_webcam_emotion(api_key):
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        # You may need to save the frame temporarily as an image file
        cv2.imwrite("temp_frame.jpg", frame)

        # Detect emotion from the captured frame
        emotions = detect_emotion(api_key, "temp_frame.jpg")

        # Display the detected emotion on the screen
        display_emotion(frame, emotions)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def display_emotion(frame, emotions):
    # Customize the display of detected emotions on the frame
    # You can use OpenCV drawing functions to overlay text on the frame
    # Example: cv2.putText(frame, f"Emotion: {pred_emotion}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Display the frame with emotion information
    cv2.imshow('Webcam Emotion Detection', frame)