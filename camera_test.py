import cv2
#Open laptop webcam
cap=cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error:webcam not detected!")
    exit()
    print("webcam started successfully!")
    print("Press Q to close the camera.")
    while True:
        ret,frame=cap.read()
        if not ret:
            print("Unable to capture video.")
            break
            cv2.imshow("SignBridge AI-Camera Test",frame)
            if cv2.waitkey(1) & 0xFF == ord('q'):
                break
                cap.release()
                cv2.destroyAllWindows()
                py camera_test.py