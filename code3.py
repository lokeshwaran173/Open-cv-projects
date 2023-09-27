import cv2

# Load the pre-trained Haar Cascade Classifier for full-body detection
full_body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')

# Open a video capture object
cap = cv2.VideoCapture('footage/footage5.mp4')

while True:
    # Read a frame from the video capture
    ret, frame = cap.read()

    if not ret:
        break

    # Convert the frame to grayscale for full-body detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Perform full-body detection
    full_bodies = full_body_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around the detected full bodies
    for (x, y, w, h) in full_bodies:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the frame with detected full bodies
    cv2.imshow('Full Body Detection', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object
cap.release()

# Close any OpenCV windows
cv2.destroyAllWindows()
