import cv2

# Initialize the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # Wait for keypress
    key = cv2.waitKey(1)

    if key == ord('q'):
        # Quit if 'q' is pressed
        break
    elif key == ord('s'):
        # Save the captured frame if 's' is pressed
        cv2.imwrite('captured_frame.jpg', frame)
        print("Photo taken!")

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()