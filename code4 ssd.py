import cv2

# Load the pre-trained Haar Cascade Classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the pre-trained SSD model for face detection
net = cv2.dnn.readNetFromCaffe('deploy.prototxt.txt', 'res10_300x300_ssd_iter_140000.caffemodel')

# Open a video capture object
cap = cv2.VideoCapture('footage/footage5.mp4')

while True:
    # Read a frame from the video capture
    ret, frame = cap.read()

    if not ret:
        break

    # Convert the frame to grayscale for Haar Cascade detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Perform Haar Cascade face detection
    faces_haar = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Perform SSD-based face detection
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1.0, size=(300, 300), mean=(104, 117, 123))
    net.setInput(blob)
    detections_ssd = net.forward()

    # Draw rectangles around the faces detected by Haar Cascade
    for (x, y, w, h) in faces_haar:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Draw rectangles around the faces detected by SSD
    for i in range(detections_ssd.shape[2]):
        confidence = detections_ssd[0, 0, i, 2]
        if confidence > 0.5:  # Adjust this threshold
            box = detections_ssd[0, 0, i, 3:7] * [frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]]
            (startX, startY, endX, endY) = box.astype(int)
            cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 2)

    # Display the frame with detected faces
    cv2.imshow('Face Detection', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object
cap.release()

# Close any OpenCV windows
cv2.destroyAllWindows()
