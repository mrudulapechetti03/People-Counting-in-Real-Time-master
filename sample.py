import cv2

cap = cv2.VideoCapture(0)  # try 1 or 2 if 0 fails
if not cap.isOpened():
    print("Camera not accessible")
else:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        cv2.imshow("Webcam Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()